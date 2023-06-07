#takes a sequence and runs it through Salis Promoter Calculator
#gets 1000 Tx_rate records as output
#saves 1 record with max and min Tx_rate values
#takes 10 records with max and min Tx_rate values
#for each -35 -10 pair finds all synonymous (AA) promoters and calculate PSSM for them
#takes top and bottom 10 promoters
#re-runs Salis' calc with those promoters (replaces hex35 + spacer + hex10 sequence in the main sequence)
#add original record info and all synonymous outputs into a csv file


import util, pickle, collections, operator, itertools, time, math, os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from Bio.Seq import Seq
import dask.dataframe as dd

#import multiprocessing

# if __name__ == "__main__":
#     from dask.distributed import Client, LocalCluster
#     cluster = LocalCluster()  # Launches a scheduler and workers locally
#     client = Client(cluster)

# k and BETA for La Fleur dataset
LOGK   = -2.80271176
BETA    = 0.81632623

def unpickler(infile):
    with open(infile, 'rb') as handle:
        obj= pickle.load(handle)
    return obj

def _revcomp(seq):
    revcomp = {'U' : 'A', 'A' : 'T', 'G' : 'C', 'T' : 'A', 'C' : 'G'}
    return "".join([revcomp[letter] for letter in seq[::-1] ])

def get_matrices(two_mer_encoder, three_mer_encoder, spacer_encoder, coeffs):

    #Extract dG values from model coefficients
    ref10_0 = coeffs.tolist()[0:64]
    ref10_3 = coeffs.tolist()[64:128]
    ref35_0 = coeffs.tolist()[128:192]
    ref35_3 = coeffs.tolist()[192:256]
    discs   = coeffs.tolist()[256:256+64]
    x10     = coeffs.tolist()[256+64:256+64+16]
    spacs   = coeffs.tolist()[256+64+16:256+64+16+3]

    # make dG matrices for each feature
    dg10_0  = util.get_dg_matrices(ref10_0,three_mer_encoder)
    dg10_3  = util.get_dg_matrices(ref10_3,three_mer_encoder)
    dg35_0  = util.get_dg_matrices(ref35_0,three_mer_encoder)
    dg35_3  = util.get_dg_matrices(ref35_3,three_mer_encoder)
    dmers   = util.get_dg_matrices(discs,three_mer_encoder)
    x10mers = util.get_dg_matrices(x10,two_mer_encoder)
    spacers = util.get_dg_matrices(spacs, spacer_encoder)

    return dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers

# Scan sequence left to right with no TSS information. Calc dG of all possible promoter configurations.
def scan_arbitrary(inputSequence, two_mer_encoder, three_mer_encoder, model, inters, constraints, dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers):
    seq_query = {}
    upstream = constraints[0]
    downstream = constraints[1]
    sequence = upstream + inputSequence + downstream

    # first 20 nt will be initial UP candidate
    for i in range(0,len(sequence)):
        tempUP = sequence[i:i+24]
        temp35 = sequence[i+25:25+i+6]  # leaves 1 nt between h35 and UPs

        # bounds defined by what was present during parameterization
        for j in range(15,21):
            tempspacer = sequence[i+25+6:25+i+6+j]
            temp10     = sequence[25+i+j+6:25+i+j+12]
            for k in range(6,11):
                tempdisc  = sequence[25+i+j+12:25+i+j+12+k]
                tempITR   =sequence[25+i+j+12+k:45+i+j+12+k]
                if len(tempITR) < 20:
                    continue
                else:
                    dG_total, dG_apparent, dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP= linear_free_energy_model(tempUP, temp35, tempspacer, temp10, tempdisc, tempITR, dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers, model, inters)
                    dG_bind  = dg10 + dg35 + dg_spacer + dg_ext10 + dg_UP
                    # dG_bind  = dg10 + dg_ext10 + dg_spacer + dg_UP
                    TSS_distance = i + len(tempUP) + len(temp35) + len(tempspacer) + len(temp10) + len(tempdisc)
                    # seq_query[(float(dG_bind), float(dG_total), TSS_distance)] = ((tempUP, temp35, tempspacer, temp10, tempdisc, tempITR),(dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP))
                    seq_query[(float(dG_total), float(dG_apparent), TSS_distance)] = ((tempUP, temp35, tempspacer, temp10, tempdisc, tempITR),(dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP))

    print ("best: ", min(seq_query.items(), key=operator.itemgetter(0)))

    best = (collections.OrderedDict(sorted(seq_query.items())), min(seq_query.items(), key=operator.itemgetter(0)))
    return best, seq_query

def linear_free_energy_model(UP, h35, spacer, h10, disc, ITR, dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers, coeffs, inters):

    prox_UP = UP[-int(len(UP)/2)::]
    dist_UP = UP[0:int(len(UP)/2)]

    # CATEGORICAL FEATURES
    ext10           = spacer[-3:-1] # TGN motif, contacts sigma
    hex10_0         = h10[0:3]
    hex10_3         = h10[3::]
    hex35_0         = h35[0:3]
    hex35_3         = h35[3::]
    disc_first_3mer = disc[0:3]
    spacer_length   = str(len(spacer))

    # NUMERICAL FEATURES
    dg_dna,dg_rna,dg_ITR     = util.calc_DNA_RNA_hybrid_energy(ITR) # calc R-loop strength
    rigidity                 = util.calc_rigidity(seq = UP + h35 + spacer[0:14])

    width_proxy_prox = util.calc_groove_width(prox_UP)
    width_proxy_dist = util.calc_groove_width(dist_UP)

    # NORMALIZE NUMERICAL FEATURES BY MAX IN TRAINING SET
    numericals         = np.array([width_proxy_dist, width_proxy_prox, dg_ITR, rigidity])
    normalizing_values = [256.0, 255.0, 4.300000000000002, 25.780434782608694]
    numerical_coefs    = np.array(coeffs.tolist()[-4::])
    normald            = np.divide(numericals,normalizing_values)
    dg_numerical       = np.multiply(normald, numerical_coefs)
    dg10      = dg10_0[hex10_0] + dg10_3[hex10_3]
    dg35      = dg35_0[hex35_0] + dg35_3[hex35_3]
    dg_disc   = dmers[disc_first_3mer]
    dg_ITR    = dg_numerical[-2]
    dg_ext10  = x10mers[ext10]

    x = float(spacer_length)
    dg_spacer = 0.1463*x**2 - 4.9113*x + 41.119

    dg_UP        = dg_numerical[0] + dg_numerical[1] + dg_numerical[-1]
    dG_apparent  = (dg10 + dg35 + dg_disc + dg_ITR + dg_ext10 + dg_spacer + dg_UP + inters[0] - LOGK)/BETA
    dG_total     = dg10 + dg35 + dg_disc + dg_ITR + dg_ext10 + dg_spacer + dg_UP + inters[0]

    return dG_total, dG_apparent, dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP

def predict(sequence, constraints):

    # Initialize model and matrices
    layer1 = np.load('free_energy_coeffs.npy')
    inters = np.load('model_intercept.npy')

    two_mer_encoder   = util.kmer_encoders(k = 2)
    three_mer_encoder = util.kmer_encoders(k = 3)
    spacer_encoder    = util.length_encoders(16, 18)
    dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers = get_matrices(two_mer_encoder = two_mer_encoder, three_mer_encoder = three_mer_encoder, spacer_encoder = spacer_encoder, coeffs = layer1)

    # Scan DNA and return predictions
    (od,result), query   = scan_arbitrary(inputSequence = sequence, two_mer_encoder = two_mer_encoder, three_mer_encoder = three_mer_encoder,
                                            model = layer1, inters = inters, constraints = constraints, dg10_0 = dg10_0, dg10_3 = dg10_3,
                                            dg35_0 =dg35_0, dg35_3 = dg35_3, dmers = dmers , x10mers = x10mers, spacers = spacers)

    dG_total, UP, h35, spacer, h10, disc, ITR = result[0][0], result[1][0][0],result[1][0][1],result[1][0][2],result[1][0][3],result[1][0][4],result[1][0][5]

    return dG_total, query, UP, h35, spacer, h10, disc, ITR

class Promoter_Calculator(object):

    def __init__(self, organism = 'Escherichia coli str. K-12 substr. MG1655',
                       sigmaLevels = {'70' : 1.0, '19' : 0.0, '24' : 0.0, '28' : 0.0, '32' : 0.0, '38' : 0.0, '54' : 0.0}):

        # Initialize model and matrices
        path = os.path.dirname(os.path.abspath(__file__))
        self.layer1 = np.load(path + '/free_energy_coeffs.npy')
        self.inters = np.load(path + '/model_intercept.npy')

        self.two_mer_encoder   = util.kmer_encoders(k = 2)
        self.three_mer_encoder = util.kmer_encoders(k = 3)
        self.spacer_encoder    = util.length_encoders(16, 18)
        self.dg10_0, self.dg10_3, self.dg35_0, self.dg35_3, self.dmers, self.x10mers, self.spacers = get_matrices(two_mer_encoder = self.two_mer_encoder, three_mer_encoder = self.three_mer_encoder, spacer_encoder = self.spacer_encoder, coeffs = self.layer1)

        self.model = self.layer1
        self.organism = organism
        self.sigmaLevels = sigmaLevels

        if organism == 'in vitro':
            self.K = 42.00000
            self.BETA = 0.81632623
        elif organism == 'Escherichia coli str. K-12 substr. MG1655':
            self.K = 42.00000
            self.BETA = 1.636217004872062
        else:
            self.K = 42.00000
            self.BETA = 1.636217004872062

    # Identify promoter with minimum dG_total (across many possible promoter states) for each TSS position in an inputted sequence.
    def predict(self, sequence, TSS_range):

        UPS_length = 24
        HEX35_length = 6
        UPS_HEX35_SPACER = 1
        SPACER_length_range = [15, 21]
        HEX10_length = 6
        DISC_length_range = [6, 11]
        ITR_length = 20

        MinPromoterSize = UPS_length + UPS_HEX35_SPACER + HEX35_length + SPACER_length_range[0] + HEX10_length + DISC_length_range[0] + ITR_length
        MaxPromoterSize = UPS_length + UPS_HEX35_SPACER + HEX35_length + SPACER_length_range[1] + HEX10_length + DISC_length_range[1] + ITR_length
        MinimumTSS = UPS_length + UPS_HEX35_SPACER + HEX35_length + SPACER_length_range[0] + HEX10_length + DISC_length_range[0]

        All_States = {}
        Min_States = {}

        #Specify fixed TSS range
        for TSS in range(TSS_range[0], TSS_range[1]):
            All_States[TSS] = {}
            for DISC_length in range(DISC_length_range[0],DISC_length_range[1]):

                if TSS - DISC_length >= 0 and TSS + ITR_length <= len(sequence):
                    tempdisc = sequence[ TSS - DISC_length : TSS  ]
                    tempITR  = sequence[ TSS : TSS + ITR_length]

                    for SPACER_length in range(SPACER_length_range[0], SPACER_length_range[1]):

                        if TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length - UPS_length - UPS_HEX35_SPACER >= 0:
                            temp10     = sequence[ TSS - DISC_length - HEX10_length : TSS - DISC_length]
                            tempspacer = sequence[ TSS - DISC_length - HEX10_length - SPACER_length : TSS - DISC_length - HEX10_length ]
                            temp35     = sequence[ TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length : TSS - DISC_length - HEX10_length - SPACER_length]
                            tempUP     = sequence[ TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length - UPS_length - UPS_HEX35_SPACER:  TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length - UPS_HEX35_SPACER]

                            dG_total, dG_apparent, dG_10, dG_35, dG_disc, dG_ITR, dG_ext10, dG_spacer, dG_UP = linear_free_energy_model(tempUP, temp35, tempspacer, temp10, tempdisc, tempITR, self.dg10_0, self.dg10_3, self.dg35_0, self.dg35_3, self.dmers, self.x10mers, self.spacers, self.model, self.inters)
                            dG_bind  = dG_10 + dG_35 + dG_spacer + dG_ext10 + dG_UP

                            Tx_rate = self.K * math.exp(- self.BETA * dG_total )

                            result = {'promoter_sequence' : sequence[TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length - UPS_length - UPS_HEX35_SPACER : TSS + ITR_length ],
                                      'TSS' : TSS, 'UP' : tempUP, 'hex35' : temp35, 'spacer' : tempspacer, 'hex10' : temp10, 'disc' : tempdisc, 'ITR' : tempITR,
                                      'dG_total' : dG_total, 'dG_10' : dG_10, 'dG_35' : dG_35, 'dG_disc' : dG_disc, 'dG_ITR' : dG_ITR, 'dG_ext10' : dG_ext10, 'dG_spacer' : dG_spacer, 'dG_UP' : dG_UP, 'dG_bind' : dG_bind,
                                      'Tx_rate' : Tx_rate,
                                      'UP_position' : tuple([TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length - UPS_HEX35_SPACER - UPS_length,
                                                       TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length - UPS_HEX35_SPACER]),
                                      'hex35_position' : tuple([TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length, TSS - DISC_length - HEX10_length - SPACER_length]),
                                      'spacer_position' : tuple([TSS - DISC_length - HEX10_length - SPACER_length, TSS - DISC_length - HEX10_length]),
                                      'hex10_position' : tuple([TSS - DISC_length - HEX10_length, TSS - DISC_length]),
                                      'disc_position' : tuple([TSS - DISC_length, TSS])
                                      }

                            All_States[TSS][ (DISC_length, SPACER_length) ] = result
                            if TSS in Min_States:
                                if result['dG_total'] < Min_States[TSS]['dG_total']:  Min_States[TSS] = result
                            else:
                                Min_States[TSS] = result

        return (Min_States, All_States)

    def run(self, sequence, TSS_range = None ):


        if TSS_range is None: TSS_range = [0, len(sequence)]

        self.sequence = sequence
        self.TSS_range = TSS_range
        self.TSS_range_rev = [len(sequence) - TSS_range[1], len(sequence) - TSS_range[0]]

        # print "self.TSS_range_rev: ", self.TSS_range_rev

        fwd_sequence = sequence
        rev_sequence = _revcomp(sequence)
        (Forward_Min_States, Forward_All_States) = self.predict(fwd_sequence, TSS_range = self.TSS_range)
        (Reverse_Min_States_Temp, Reverse_All_States_Temp) = self.predict(rev_sequence, TSS_range = self.TSS_range_rev)

        Reverse_Min_States = {}
        Reverse_All_States = {}
        # 0  <------>500 fwd 500 bp
        # 500<------>0   rev 500 bp
        #      275 TSS
        #      200-300
        #500-275 = 225
        #
        for TSS in Reverse_Min_States_Temp.keys():
            Reverse_Min_States[len(sequence) - TSS] = Reverse_Min_States_Temp[TSS]
            Reverse_All_States[len(sequence) - TSS] = Reverse_All_States_Temp[TSS]

        self.Forward_Predictions_per_TSS = Forward_Min_States
        self.Reverse_Predictions_per_TSS = Reverse_Min_States

    def output(self):
        output = {'organism' : self.organism,
                  'sigmaLevels' : self.sigmaLevels,
                  'K' : self.K,
                  'beta' : self.BETA,
                  'sequence' : self.sequence,
                  'TSS_range' : self.TSS_range,
                  'Forward_Predictions_per_TSS' : self.Forward_Predictions_per_TSS,
                  'Reverse_Predictions_per_TSS' : self.Reverse_Predictions_per_TSS
                }

        df_fwd = pd.DataFrame.from_dict(output['Forward_Predictions_per_TSS'], orient = 'index')
        df_rev = pd.DataFrame.from_dict(output['Reverse_Predictions_per_TSS'], orient = 'index')
        return df_fwd, df_rev

    # Scan sequence left to right with no TSS information. Calc dG of all possible promoter configurations. Return promoter with minimum dG_total.
    def oldScan(self, inputSequence, preSeq, postSeq):
        seq_query = {}
        sequence = preSeq + inputSequence + postSeq

        #print "sequence: ", sequence

        # first 20 nt will be initial UP candidate
        for i in range(0,len(sequence)):
            tempUP = sequence[i:i+24]
            temp35 = sequence[i+25:25+i+6]  # leaves 1 nt between h35 and UPs

            # bounds defined by what was present during parameterization
            for j in range(15,21):
                tempspacer = sequence[i+25+6:25+i+6+j]
                temp10     = sequence[25+i+j+6:25+i+j+12]
                for k in range(6,11):
                    tempdisc  = sequence[25+i+j+12:25+i+j+12+k]
                    tempITR   =sequence[25+i+j+12+k:45+i+j+12+k]
                    if len(tempITR) < 20:
                        continue
                    else:
                        dG_total, dG_apparent, dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP= linear_free_energy_model(tempUP, temp35, tempspacer, temp10, tempdisc, tempITR, self.dg10_0, self.dg10_3, self.dg35_0, self.dg35_3, self.dmers, self.x10mers, self.spacers, self.model, self.inters)
                        dG_bind  = dg10 + dg35 + dg_spacer + dg_ext10 + dg_UP
                        # dG_bind  = dg10 + dg_ext10 + dg_spacer + dg_UP
                        TSS_distance = i + len(tempUP) + len(temp35) + len(tempspacer) + len(temp10) + len(tempdisc)
                        # seq_query[(float(dG_bind), float(dG_total), TSS_distance)] = ((tempUP, temp35, tempspacer, temp10, tempdisc, tempITR),(dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP))
                        seq_query[(float(dG_total), float(dG_apparent), TSS_distance)] = ((tempUP, temp35, tempspacer, temp10, tempdisc, tempITR),(dg10, dg35, dg_disc, dg_ITR, dg_ext10, dg_spacer, dg_UP))

        best = (collections.OrderedDict(sorted(seq_query.items())), min(seq_query.items(), key=operator.itemgetter(0)))
        return best, seq_query


def process_promoters_nt(promoters_35, promoters_10, spacers_lst):

    #locus_tag_ids = list()

    prom_lst = {'Promoters_35': promoters_35, 'Promoters_10': promoters_10, 'Spacers': spacers_lst}

    df = pd.DataFrame(prom_lst)
    #split promoters sequences into characters
    p_35 = df['Promoters_35']
    p_10 = df['Promoters_10']
    spacers = df['Spacers']

    print('Splitting promoters to nt')
    #df_35 = pd.DataFrame(p_35.str.split('',6).tolist())
    df_35 = pd.DataFrame(p_35.str.split('',6).tolist())
    df_10 = pd.DataFrame(p_10.str.split('',6).tolist())

    df_35 = df_35.drop(df_35.columns[0], axis = 1)
    df_10 = df_10.drop(df_10.columns[0], axis = 1)



    #caculate PSSM score for each promoter and add into a column
    df_35 = calculate_pssm(df_35, '35')
    #df_35['Locus_ID'] = prom_lst['Locus_ID']
    df_10 = calculate_pssm(df_10, '10')

    df_35 = pd.concat([pd.DataFrame(p_35), df_35], axis=1)
    df_10 = pd.concat([pd.DataFrame(p_10), df_10], axis=1)

    pssm_df = pd.concat([df_35, df_10, spacers], axis=1)



    #Find the smallest and largest values
    print('Number of records (total) = ' + str(len(df_35)))
    print('PSSM 35 max = ' + str(df_35['PSSM_Score_35'].max()))
    print('PSSM 35 min = ' + str(df_35['PSSM_Score_35'].min()))
    print('PSSM 10 max = ' + str(df_10['PSSM_Score_10'].max()))
    print('PSSM 10 min = ' + str(df_10['PSSM_Score_10'].min()))


    #save to CSV
    #df_35.to_csv('PSSM_all_promoters_35.csv')
    #Sdf_10.to_csv('PSSM_all_promoters_10.csv')

    #analyze nucleotide frequencies
    calc_frequencies(df_35, '35')
    calc_frequencies(df_10, '10')
    #print(p_35_series)
    #print(p_35_series.value_counts(normalize=True))



    return pssm_df

def calc_PSSM(row, frame, type_str):

    # Opening JSON dictionaries
    if type_str == '35':
        pssm_dic = json.load(open('pssm_35.json'))['PSSM_35']
    if type_str == '10':
        pssm_dic = json.load(open('pssm_10.json'))['PSSM_10']

    pssm_val = 0
    frame_row = row
    if frame != 0:
        frame_row = row[frame:6 + frame]
    for i, nt in enumerate(frame_row):
        pssm_index_str = nt + str(i + 1)
        # print(pssm_index_str)
        pssm_val += pssm_dic[pssm_index_str]

    return round(pssm_val,2)

def calc_frequencies(df, prom_type):
    #freq_df = df['1', '2', '3', '4', '5', '6'].copy()
    freq_df = df.iloc[:, [0, 1, 2, 3, 4, 5]].copy()

    freq_result_df = pd.DataFrame()
    for col_num in freq_df:
        #print('Column ' + str(col_num))
        #print(df[df.columns[idx]].value_counts(normalize=True))
        freq_series = freq_df[col_num].value_counts(normalize=True)
        #print(freq_series)
        freq_result_df = freq_result_df.append(freq_series)

    freq_result_df.to_csv('nt_frequencies for ' + prom_type + '.csv')
    #print(freq_result_df)

def calculate_pssm(df, type_str):
    pssm_all_lst = list()

    print('PSSM calculation')

    #replace for
    PSSM_values = df.apply(
        lambda row: calc_PSSM(row, type_str),
        axis = 1)

    df['PSSM_Score_' + type_str] =  PSSM_values
    return df

def findAllPromoterAAPermutations(aa_promoter, aa_dic_df, type, frame, tx_rate_df, tss):

    firstAA = aa_promoter[0]
    secondAA = aa_promoter[1]
    if len(aa_promoter) == 3:
        thirdAA = aa_promoter[2]

    #print(aa_promoter)
    firstAA_codons = aa_dic_df.loc[aa_dic_df['shortCode'] == firstAA]['codons']
    firstAA_codons = firstAA_codons.iloc[0]

    secondAA_codons = aa_dic_df.loc[aa_dic_df['shortCode'] == secondAA]['codons']
    secondAA_codons = secondAA_codons.iloc[0]

    if len(aa_promoter) == 3:
        thirdAA_codons = aa_dic_df.loc[aa_dic_df['shortCode'] == thirdAA]['codons']
        thirdAA_codons = thirdAA_codons.iloc[0]

    #combine AA first and second, find their PSSM and return
    perm_prom_list = list()
    for first in firstAA_codons:
        for second in secondAA_codons:
            prom = first + second
            ##if prom != aa_promoter:
                #exclude the promoter that is being processed
            if len(aa_promoter) == 3:
                for third in thirdAA_codons:
                    prom = first + second + third
                    perm_prom_list.append(prom)
            else:
                perm_prom_list.append(prom)

    perm_prom_pssm_df = pd.DataFrame()
    perm_prom_pssm_df['Promoters_perm_nt'] = perm_prom_list
    #calculate_pssm
##    perm_prom_pssm_df['PSSM_Promoters_perm'] = perm_prom_pssm_df['Promoters_perm_nt'].apply(lambda x: calc_PSSM(x, type))
    #calc PSSM. for 3 aa need to use frame to get the permutations for the original sequence
    #calculate_pssm

    perm_prom_pssm_df['PSSM_Promoters_perm'] = perm_prom_pssm_df.apply(lambda x: calc_PSSM(x['Promoters_perm_nt'], frame, type), axis = 1)
    perm_prom_pssm_df = perm_prom_pssm_df.reindex()
    perm_prom_pssm_df = perm_prom_pssm_df.sort_values(by = 'PSSM_Promoters_perm', ascending = False)
    #add AA sequences
    perm_prom_pssm_df['Promoters_perm_aa'] = perm_prom_pssm_df['Promoters_perm_nt'].apply(lambda x: str(Seq(x).translate(table = "Bacterial")))
    perm_prom_pssm_df['TSS'] = tss
    perm_prom_pssm_df['frame'] = frame

##    perm_prom_pssm_df = perm_prom_pssm_df.reindex()
##    perm_prom_pssm_df = perm_prom_pssm_df.sort_values(by = 'PSSM_Promoters_perm', ascending = False)

    return perm_prom_pssm_df

# frame shift for the promoters: 0, 1, 2 - the remainder from the division
#-35 and -10 6nt might be in different reading frames
def find_frame_shift(gene_sequence, promoter_sequence, UP, hex35, spacer, hex10):

    #find promoter sequence
    pos_nt = gene_sequence.find(promoter_sequence)

    #get a sequence from the beginning till the end of the promoter
    #cannot just search for hex35+spacer+hex10 - might be not unique
    gene_sequence_prom = gene_sequence[0:pos_nt+len(promoter_sequence)]

    pos_nt_35 = gene_sequence_prom.find(hex35+spacer+hex10)
    frame_35 = pos_nt_35%3
    pos_nt_10 = pos_nt_35 + len(hex35) + len(spacer)
    frame_10 = pos_nt_10 % 3

    return pd.Series({"frame35": frame_35,"frame10": frame_10})

def recalculate_aa_promoter_sequence(promoter_sequence, hex35, spacer, hex10, frame35, frame10, gene_sequence):

    new_promoter_35 = hex35
    new_promoter_10 = hex10
    new_spacer = spacer
    full_prom_str = hex35+spacer+hex10
    prom_pos = promoter_sequence.find(full_prom_str)
    spacer_pos = promoter_sequence.find(spacer)
    #pos without UP

    if frame35 != 0:
        new_pos35 = prom_pos - frame35
        new_promoter_35 = promoter_sequence[new_pos35: new_pos35 + 9]
    if frame10 !=0:
        new_pos35 = prom_pos - frame35
        new_pos10 = new_pos35 + len(hex35) + len(spacer) + frame35 - frame10
        new_promoter_10 = promoter_sequence[new_pos10: new_pos10  + 9]
            #new_promoter = str(Seq(new_promoter).translate(table = "Bacterial"))

    new_promoter_35_aa = str(Seq(new_promoter_35).translate(table = "Bacterial"))
    new_promoter_10_aa = str(Seq(new_promoter_10).translate(table = "Bacterial"))
    return pd.Series({'recalc_AA_promoter_35':new_promoter_35_aa, 'recalc_AA_promoter_10': new_promoter_10_aa, 'recalc_nt_promoter_35': new_promoter_35, 'recalc_spacer': new_spacer, 'recalc_nt_promoter_10':new_promoter_10})

def process_promoters_aa(df, tx_rate_df, direction_type):

    aa_dic_json = json.load(open('pssm_aa_table.json'))
    #aa_dic = pd.read_json('PSSM_aa_table.json')
    aa_dic_df = pd.json_normalize(aa_dic_json['aminoAcids'])

    #find if there is a frame shift
    if direction_type == 'fwd':
        sequence = tx_rate_df['sequence']
    if direction_type == 'rev':
        sequence = tx_rate_df['sequence_compl']

    #update AA values for permutations based on the frame
    #need to add 1 nt before and 2 after
    frame_df = pd.DataFrame()
    frame_df = df.apply(lambda x: find_frame_shift(sequence, x['promoter_sequence'], x['UP'], x['hex35'], x['spacer'], x['hex10']), axis = 1)
    df = pd.concat([df, frame_df], axis = 1)
    ##df[['recalc_AA_promoter_35'],['recalc_AA_promoter_10']] = df.apply(lambda x: recalculate_aa_promoter_sequence(x['UP'], x['hex35'], x['spacer'], x['hex10'], x['frame'], sequence), axis = 1)
    df_recalc = df.apply(lambda x: recalculate_aa_promoter_sequence(x['promoter_sequence'], x['hex35'], x['spacer'], x['hex10'], x['frame35'], x['frame10'], sequence), axis = 1)
    df = pd.concat([df, df_recalc], axis = 1)



    #print(aa_dic_df.loc[aa_dic_df['shortCode'] == 'A']['codons'])
    ##df_35_perm_prom = df['AA_Promoter_35'].apply(lambda x: findAllPromoterAAPermutations(x, aa_dic_df, '35', tx_rate_df))
    df_35_perm_prom = df.apply(lambda x: findAllPromoterAAPermutations(x['recalc_AA_promoter_35'], aa_dic_df, '35', x['frame35'], tx_rate_df, x['TSS']), axis = 1)
    df_35_perm_prom = pd.concat(df_35_perm_prom.tolist())
    df_35_perm_prom = df_35_perm_prom.rename(columns={'frame': 'frame35'})


    df_10_perm_prom = df.apply(lambda x: findAllPromoterAAPermutations(x['recalc_AA_promoter_10'], aa_dic_df, '10', x['frame10'], tx_rate_df, x['TSS']), axis = 1)
    df_10_perm_prom = pd.concat(df_10_perm_prom.tolist())
    df_10_perm_prom = df_10_perm_prom.rename(columns={'frame': 'frame10'})


    df_35_perm_prom = df_35_perm_prom.drop_duplicates()
    df_10_perm_prom = df_10_perm_prom.drop_duplicates()

    return df_35_perm_prom, df_10_perm_prom


#row has the original promoter
def run_salis_calc(row, row_5, original_prom_sequence, dir_type, range, tx_rate_df):

    #restore original 9 nt promoter and spacer if frames are not 0 to replace it with new 9 nt promoters
    frame35 = row_5['frame35']
    frame10 = row_5['frame10']
    spacer = row['spacer']
    prev_hex35 = row['hex35']
    prev_hex10 = row['hex10']
    row_5_hex35 = row_5['hex35']
    row_5_hex10 = row_5['hex10']
    new_hex35_6nt = row_5['hex35']
    new_hex10_6nt = row_5['hex10']

    old_prom_sequence_hex = prev_hex35 + str(row['spacer']) + prev_hex10
    pos_35 = original_prom_sequence.find(old_prom_sequence_hex)

    #get 6nt from new -35 and 10 promoters if there're frames
    if frame35 != 0:
        if frame35 == 2:
            new_hex35_6nt = new_hex35_6nt[2:2+6]
        if frame35 == 1:
            new_hex35_6nt = new_hex35_6nt[1:1+6]

    if frame10 != 0:
        if frame10 == 2:
            new_hex10_6nt = new_hex10_6nt[2:2+6]
        if frame10 == 1:
            new_hex10_6nt = new_hex10_6nt[1:1+6]


    prev_hex35 = row['hex35_9nt']
    prev_hex10 = row['hex10_9nt']
    spacer = row['spacer_9nt']
    prev_prom_seq = prev_hex35+spacer+prev_hex10
    new_prom_seq = row_5_hex35+spacer+row_5_hex10

    new_promoter_seq = original_prom_sequence.replace(prev_prom_seq, new_prom_seq)



    #aa sequence of promoters do not match here because of the frame!

    ##new_prom_sequence_hex = row_5['hex35'] + str(row['spacer']) + row_5['hex10']
    ##old_prom_sequence_hex = row['hex35'] + str(row['spacer']) + row['hex10']
    sequence = tx_rate_df["sequence"]
    sequence_compl = tx_rate_df["sequence_compl"]

    # if(dir_type == 'fwd'):
    #     new_sequence = sequence.replace(original_prom_sequence, new_prom_sequence)
    # if(dir_type == 'rev'):
    #     original_prom_sequence =
    #     new_sequence = sequence.replace(original_prom_sequence, new_prom_sequence)

    if(dir_type == 'fwd'):
        ##new_prom_sequence = original_prom_sequence.replace(old_prom_sequence_hex, new_prom_sequence_hex)
        new_sequence = sequence.replace(prev_prom_seq, new_prom_seq)
    if(dir_type == 'rev'):
        ##new_prom_sequence = original_prom_sequence.replace(old_prom_sequence_hex, new_prom_sequence_hex)
        new_sequence = sequence_compl.replace(prev_prom_seq, new_prom_seq)
        sequence = sequence_compl

    new_sequence_aa = str(Seq(new_sequence).translate(table = "Bacterial"))
    sequence_aa = str(Seq(sequence).translate(table = "Bacterial"))

    # if(new_sequence_aa != sequence_aa):
    #     print("ERROR: sequences are not matching, TSS =" + str(row['TSS']))
    # else:
    #     print("TSS = " + str(row['TSS']) + " matching")

    TSS_new_res = pd.DataFrame()


    calc = Promoter_Calculator()
    calc.run(new_sequence, TSS_range=[0, len(new_sequence)])
    fwd_new_res, rev_new_res = calc.output()
    end_time = datetime.now()
    ##print('Duration of 1 salis calc run (util): {}'.format(end_time - start_time))


    if range == 'max':
        #TSS_new_res = TSS_new_res.loc[TSS_new_res['Tx_rate'].astype(float) >= float(tx_rate_df['max_fwd'])]
        #TSS_new_res = TSS_new_res.loc[TSS_new_res['Tx_rate'].astype(float) < float(tx_rate_df['max_fwd'])]
        TSS_new_res = fwd_new_res.loc[fwd_new_res['Tx_rate'].astype(float) > float(row['Tx_rate'])]
    if range == 'min':
        #TSS_new_res = TSS_new_res.loc[TSS_new_res['Tx_rate'].astype(float) <= float(tx_rate_df['min_fwd'])]
        #TSS_new_res = TSS_new_res.loc[TSS_new_res['Tx_rate'].astype(float) > float(tx_rate_df['min_fwd'])]
        TSS_new_res = fwd_new_res.loc[fwd_new_res['Tx_rate'].astype(float) < float(row['Tx_rate'])]

    # filter by substitution promoters only + ITR
    #compare 6nt and the original spacer
    #match_TSS_df = TSS_new_res.loc[TSS_new_res['hex35'] == row_5['hex35']]
    match_TSS_df = TSS_new_res.loc[TSS_new_res['hex35'] == new_hex35_6nt]
    #match_TSS_df = match_TSS_df.loc[TSS_new_res['hex10'] == row_5['hex10']]
    match_TSS_df = match_TSS_df.loc[TSS_new_res['hex10'] == new_hex10_6nt]
    match_TSS_df = match_TSS_df.loc[TSS_new_res['ITR'] == row['ITR']]
    ##match_TSS_df = match_TSS_df.loc[TSS_new_res['UP'] == row['UP']]
    match_TSS_df = match_TSS_df.loc[TSS_new_res['spacer'] == row['spacer']]

    #if no matching records, then the promoters are silenced
    if len(match_TSS_df) == 0 and range == 'min':
        match_TSS_df = row.to_frame().T
        match_TSS_df['Tx_rate'] = 0.1
        match_TSS_df['hex35'] = new_hex35_6nt
        match_TSS_df['hex10'] = new_hex10_6nt
        ##print("NO MATCHES")

    match_TSS_df['Type'] = 'Modified Promoter'
    match_TSS_df['direction'] = dir_type
    ###match_TSS_df['AA_Promoter_35'] = match_TSS_df['hex35'].apply(lambda x: str(Seq(x).translate()))
    ###match_TSS_df['AA_Promoter_10'] = match_TSS_df['hex10'].apply(lambda x: str(Seq(x).translate()))
    match_TSS_df = match_TSS_df.sort_values(by=['Tx_rate'], ascending=False)
    match_TSS_df['ID'] = row['TSS']
    match_TSS_df['TSS'] = row['TSS']
    match_TSS_df['new_gene_sequence'] = new_sequence
    match_TSS_df['hex35_9nt'] = row_5_hex35
    match_TSS_df['hex10_9nt'] = row_5_hex10

    ##print("hex 35 = " + match_TSS_df['hex35'] + " , hex10 = " + match_TSS_df['hex10'])
    match_TSS_df['AA_Promoter_35'] = match_TSS_df['hex35'].apply(lambda x: str(Seq(x).translate(table = "Bacterial")))
    match_TSS_df['AA_Promoter_10'] = match_TSS_df['hex10'].apply(lambda x: str(Seq(x).translate(table = "Bacterial")))

    #add frames
    match_TSS_df['frame'] = str(int(frame35) + 1)

    return match_TSS_df

def get_9nt_promoter(frame35, frame10, hex35, hex10, spacer, sequence):

    prom_sequence_hex = hex35 + spacer + hex10
    pos_35 = sequence.find(prom_sequence_hex)
    new_spacer = spacer

    prev_hex35_9nt = hex35
    prev_hex10_9nt = hex10
    if frame35 != 0:
        if frame35 == 2:
            new_spacer = spacer[ 1:]
        if frame35 == 1:
            new_spacer = spacer[ 2:]

        prev_hex35_9nt = sequence[pos_35-frame35:pos_35-frame35 + 9]
        prev_hex35 = prev_hex35_9nt

    if frame10 != 0:
        if frame10 == 2:
            new_spacer = spacer[:-1]
        if frame10 == 1:
            new_spacer = spacer[:-2]

        pos_10 = pos_35 + len(hex35) + len(spacer) - frame10
        prev_hex10_9nt = sequence[pos_10: pos_10 + 9]

    return prev_hex35_9nt, new_spacer, prev_hex10_9nt
def match_promoters(row, df_35_perm_prom, df_10_perm_prom, dir_type, range, tx_rate_df):
    #print(row)

    TSS_res_df = pd.DataFrame()
    match_35_promoters_df = df_35_perm_prom.loc[df_35_perm_prom['TSS'].values == row['TSS']]
    match_10_promoters_df = df_10_perm_prom.loc[df_10_perm_prom['TSS'].values == row['TSS']]


    match_35_promoters_df = match_35_promoters_df.rename(columns={'Promoters_perm_nt': 'hex35'})
    match_35_promoters_df = match_35_promoters_df.rename(columns={'PSSM_Promoters_perm': 'PSSM_Promoters_perm_35'})
    match_35_promoters_df = match_35_promoters_df.rename(columns={'Promoters_perm_aa': 'PSSM_Promoters_perm_aa_35'})

    match_10_promoters_df = match_10_promoters_df.rename(columns={'Promoters_perm_nt': 'hex10'})
    match_10_promoters_df = match_10_promoters_df.rename(columns={'PSSM_Promoters_perm': 'PSSM_Promoters_perm_10'})
    match_10_promoters_df = match_10_promoters_df.rename(columns={'Promoters_perm_aa': 'PSSM_Promoters_perm_aa_10'})

    # top_promoters_df = pd.concat([match_35_promoters_df_5, match_35_promoters_df_5], ignore_index=True, axis = 1)
    top_promoters_df = match_35_promoters_df.merge(match_10_promoters_df, on=["TSS"])
    top_promoters_df = top_promoters_df[['TSS', 'frame35', 'frame10', 'hex35', 'hex10', 'PSSM_Promoters_perm_10', 'PSSM_Promoters_perm_35']].copy()
    top_promoters_df = top_promoters_df.drop_duplicates()

    #specify how many combinations of promoters will be run: min = 20 (by default) or all (might take long time)
    if range == 'max':
        if tx_rate_df['prom_opt'] == 'min':
            top_promoters_df = top_promoters_df.head(20)
        else:
            top_promoters_df = top_promoters_df
    if range == 'min':
        if tx_rate_df['prom_opt'] == 'min':
            top_promoters_df = top_promoters_df.tail(20)
        else:
            top_promoters_df = top_promoters_df

    # ORIGINAL VALUES
    original_prom_sequence = str(row['promoter_sequence'])
    original_TSS = row['Tx_rate']

    row['Type'] = 'Original Promoter'
    row['direction'] = dir_type
    row['ID'] = row['TSS']
    if(dir_type == 'fwd'):
        row['new_gene_sequence'] = tx_rate_df['sequence']
    elif dir_type == 'rev':
        row['new_gene_sequence'] = tx_rate_df['sequence_compl']

    row_frame = find_frame_shift(row['new_gene_sequence'], row['promoter_sequence'], row['UP'], row['hex35'], row['spacer'], row['hex10'])
    row['frame35'] = row_frame['frame35']
    row['frame10'] = row_frame['frame10']
    row['frame'] = str(row['frame35'] + 1)

    new_hex35, new_spacer, new_hex10 =  get_9nt_promoter(row_frame['frame35'], row_frame['frame10'], row['hex35'], row['hex10'], row['spacer'], row['new_gene_sequence'])
    row['hex35_9nt'] = new_hex35
    row['hex10_9nt'] = new_hex10
    row['spacer_9nt'] = new_spacer

    TSS_res_df = pd.concat([TSS_res_df, row.to_frame().T], ignore_index=True, axis = 0)

    # SUBSTITUTIONS
    # TODO: optimise
    #dask_top_promoters_df = dd.from_pandas(top_promoters_df, npartitions=30)
    dask_top_promoters_df = dd.from_pandas(top_promoters_df, npartitions=30)
    TSS_res_promoters_df = dask_top_promoters_df.map_partitions(lambda df: df.apply(lambda x: run_salis_calc(row, x, original_prom_sequence, dir_type, range, tx_rate_df), axis=1), meta=pd.Series(dtype='object')).compute()

    ##TSS_res_promoters_df = top_promoters_df.apply(lambda x: run_salis_calc(row, x, original_prom_sequence, dir_type, range, tx_rate_df), axis=1)
    TSS_res_promoters_df = pd.concat(TSS_res_promoters_df.tolist())

    #filter by ITR
    ##TSS_res_promoters_df = TSS_res_promoters_df.loc[TSS_res_promoters_df['ITR'] == row['ITR']]

    ###TSS_res_promoters_df = TSS_res_promoters_df.loc[TSS_res_promoters_df['ITR'].values == row['ITR']]


    #match_TSS_df['AA_Promoter_35'] = match_TSS_df['hex35'].apply(lambda x: str(Seq(x).translate()))
    #match_TSS_df['AA_Promoter_10'] = match_TSS_df['hex10'].apply(lambda x: str(Seq(x).translate()))

    #TSS_res_df = TSS_res_df.append(TSS_res_promoters_df)
    TSS_res_df = pd.concat([TSS_res_df, TSS_res_promoters_df])

    return TSS_res_df


def substitute_promoters(TSS_top_df, df_35_perm_prom, df_10_perm_prom, dir_type, range, tx_rate_df):

    #filer 35 and 10 by AA_Promoter_35, AA_Promoter_10 - aa sequence match
    #for each row find top values for 35 and 10 they should be > than the original values PSSM_Score_10 and PSSM_Score_35

    #or do the substitution for all promoters pair and choose top 20 improved TSS
    TSS_res_df = TSS_top_df.apply(lambda x: match_promoters(x, df_35_perm_prom, df_10_perm_prom, dir_type, range, tx_rate_df), axis = 1)
    TSS_res_df = pd.concat(TSS_res_df.tolist())

    # remove promoters that change the original AA sequence
    if dir_type == 'fwd':
        aa_orig_sequence = str(Seq(tx_rate_df['sequence']).translate(table = "Bacterial"))
    elif dir_type == 'rev':
        aa_orig_sequence = str(Seq(tx_rate_df['sequence_compl']).translate(table = "Bacterial"))

    TSS_res_df['aa_new_gene_sequence'] = TSS_res_df['new_gene_sequence'].apply(lambda x: str(Seq(x).translate(table = "Bacterial")))
    TSS_res_df = TSS_res_df.reset_index(drop=True)
    TSS_res_df = TSS_res_df.drop(TSS_res_df[TSS_res_df.aa_new_gene_sequence != aa_orig_sequence].index)

    return TSS_res_df

def TSS_results_to_df(result):
    tss_df = pd.DataFrame.from_dict(result, orient = 'index')
    #tss_df = pd.DataFrame(result)
    #print(result[1])
    #tss_df[1] = tss_df.apply(literal_eval)
    #tss_df = tss_df.join(pd.json_normalize(tss_df))

    #tss_df2 = pd.DataFrame.from_dict(result, orient='index',
    #                       columns=['Tx_rate', 'promoter_sequence', 'UP', 'hex35', 'hex10'])

    return tss_df

def process_TSS_results(TSS, result):

    #tss_df = pd.DataFrame.from_dict(result, orient='index', columns = ['TSS', 'Tx_rate', 'promoter_sequence'])
    #tss_df = pd.DataFrame()
    record = {'TSS': TSS, 'Tx_rate': result['Tx_rate'], 'promoter_sequence':result['promoter_sequence'], 'UP': result['UP'], 'hex35': result['hex35'],
              'hex10': result['hex10'], 'spacer': result['spacer'], 'disc': result['disc'], 'ITR': result['ITR'], 'dG_total': result['dG_total'],
              'dG_10': result['dG_10'], 'dG_35': result['dG_35'], 'dG_disc': result['dG_disc'], 'dG_ITR': result['dG_ITR'], 'dG_ext10': result['dG_ext10'],
              'dG_spacer': result['dG_spacer'], 'dG_UP': result['dG_UP'], 'dG_bind': result['dG_bind']
              , 'UP_position':tuple(result['UP_position'])
              , 'hex35_position': tuple(result['hex35_position'])
              , 'spacer_position': tuple(result['spacer_position'])
              , 'hex10_position': tuple(result['hex10_position'])
              , 'disc_position': tuple(result['disc_position'])
              }
    #tss_df = pd.DataFrame(record)
    tss_df = pd.DataFrame.from_dict([record])

    #filter out stop codons in -10 and -35 promoters
    tss_df = tss_df[tss_df['hex35'].find('')]
    tss_df = tss_df['*' not in tss_df['hex10']]

    #tss_df1 = tss_df.from_dict(tss_df[])


    return tss_df


def process_df_promoters(df, direction_type, type, tx_rate_df):
    #deal with synonymous codon changes
    # we need to add all synonymous codon combination for -35 and -10, then calculate their PSSM
    df_35_perm_prom, df_10_perm_prom = process_promoters_aa(df, tx_rate_df, direction_type)

    df_35_perm_prom = df_35_perm_prom.drop_duplicates()
    df_10_perm_prom = df_10_perm_prom.drop_duplicates()

    df_35_perm_prom = df_35_perm_prom.sort_values(by=['PSSM_Promoters_perm'], ascending=False)
    df_10_perm_prom = df_10_perm_prom.sort_values(by=['PSSM_Promoters_perm'], ascending=False)

    res_df = substitute_promoters(df, df_35_perm_prom, df_10_perm_prom, direction_type, type, tx_rate_df)

    #rename ID to Parent_ID column
    res_df = res_df.rename(columns={'ID': 'Parent_ID'})
    return res_df

def calc_tx_rate_fold(df, res_row, range):
    tx_rate_fold = 1
    if res_row['Type'] == "Modified Promoter":
        TSS_val = res_row['TSS']
        parent_txt_rate_df = df.loc[(df['TSS'] == TSS_val) & (df['Type'] == 'Original Promoter')]
        parent_tx_rate = parent_txt_rate_df['Tx_rate'].values[0]
        child_tx_rate = res_row['Tx_rate']

        if range == 'max':
            tx_rate_fold =  round((child_tx_rate/parent_tx_rate),2)
        if range == 'min':
            tx_rate_fold = -abs(round((parent_tx_rate/child_tx_rate), 2))

    return tx_rate_fold

#calculate the fold change base on a parent TSS
def add_txrate_foldchange_col(df, range):
    df_copy = df.copy()
    #df_copy['Tx_rate_FoldChange'] = df_copy['Tx_rate']/rate.astype(float)
    #df_copy['Tx_rate_FoldChange'] = df_copy['Tx_rate'].apply(lambda x: x.astype(float).round(2))
    #df_copy['Tx_rate_FoldChange'] =  df_copy['Tx_rate_FoldChange'].astype(float).round(2)

    df_copy['Tx_rate_FoldChange'] = df_copy.apply(lambda x: calc_tx_rate_fold(df, x, range), axis = 1)
    return df_copy
