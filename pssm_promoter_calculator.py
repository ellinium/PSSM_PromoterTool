import pandas as pd
from datetime import datetime
from Bio.Seq import Seq
from Bio import SeqIO
import sys
#For colab
#from google.colab import files
##sys.path.append(os.path.abspath('PSSM_PromoterTool/'))
import pssm_util

OUTPUT_FILE_NAME = "PSSMPromoterCalculator"
USE_PROMOTERS_OUTPUT = "using the synonymous promoters "
def get_gene_sequence():
    gene_sequence = ""
    if len(sys.argv) < 2:
        raise Exception('Please provide a file name with a gene sequence.')

    gene_file_name = sys.argv[1]

    f = open(gene_file_name, "r")
    f_content = f.read()

    if '>' in f_content:
        records = list(SeqIO.parse(gene_file_name, "fasta"))
        gene_sequence = str(records[0].seq)
    else:
        gene_sequence = f_content

    print("processing " + gene_file_name)
    return str.upper(gene_sequence)

#number of promoter combinations to run: top 20 - by default and all - to run all combinations
def get_promoter_opt():
    if len(sys.argv) == 3:
        prom_opt = str.lower(sys.argv[2])
        if (prom_opt != 'all'):
            raise Exception(
                'Unrecognised value for the promoter combinations. Please specify "''all''" for all combinations.')
    else:
        prom_opt = 'min'

    return prom_opt

if __name__ == "__main__":

    from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster()  # Launches a scheduler and workers locally
    # client = Client(cluster)
    client = Client()
    # import dask
    # dask.config.set(scheduler='processes')
    start_time = datetime.now()

    #sequence = "".join([random.choice(['A','G','C','T']) for x in range(1000)])
    #sequence = "".join([random.choice(['A','G','C','T']) for x in range(100)])

    #sequence = "TCTATGCTCCAGGGCGATTAGGGAACAGCGTGTTGCTGGTCAGTAGTGTACCCTAGCCCACATAGCTACTTTTACTTCGTCCGTTCAGCGGACAAACGCT"
    #sequence = "ATGGTACGCTGGACTTTGTGGGATACCCTCGCTTTCCTGCTCCTGTTGAGTTTATTGCTGCCGTCATTGCTTATTATGTTCATCCCGTCAACATTCAAACGGCCTGTCTCATCATGGAAGGCGCTGAATTTACGGAAAACATTATTAATGGCGTCGAGCGTCCGGTTAAAGCCGCTGAATTGTTCGCGTTTACCTTGCGTGTACGCGCAGGAAACACTGACGTTCTTACTGACGCAGAAGAAAACGTGCGTCAAAAATTACGTGCGGAAAGAATGA"
    #sequence = "ATGAGTCAAGTTACTGAACAATCCGTACGTTTCCAGACCGCTTTGGCCTCTATTAAGCTCATTCAGGCTTCTGCCGTTTTGGATTTAACCGAAGATGATTTCGATTTTCTGACGAGTAACAAAGTTTGGATTGCTACTGACCGCTCTCGTGCTCGTCGCTGCGTTGAAGCTTGCGTTTACGGTACGCTGGACTTTGTGGGATACCCTCGCTTTCCTGCTCCTGTTGAGTTTATTGCTGCCGTCATTGCTTATTATGTTCATCCCGTCAACATTCAAACGGCCTGTCTCATCATGGAAGGCGCTGAATTTACGGAAAACATTATTAATGGCGTCGAGCGTCCGGTTAAAGCCGCTGAATTGTTCGCGTTTACCTTGCGTGTACGCGCAGGAAACACTGACGTTCTTACTGACGCAGAAGAAAACGTGCGTCAAAAATTACGTGCGGAGGGTGTGATGTAA"
    #sequence = "atgtctaaaggtaaaaaacgttctggcgctcgccctggtcgtccgcagccgttgcgaggtactaaaggcaagcgtaaaggcgctcgtctttggtatgtaggtggtcaacaattttaa"


    sequence = get_gene_sequence()

    calc = pssm_util.Promoter_Calculator()
    calc.run(sequence, TSS_range = [0, len(sequence)])
    rev_TSS_df = calc.output()
    fwd_TSS_df, rev_TSS_df = calc.output()

    #fwd_TSS_df = fwd_TSS_df.drop_duplicates(subset=['hex35', 'hex10', 'spacer', 'ITR'], keep='last')
    fwd_TSS_df = fwd_TSS_df.drop_duplicates(subset=['hex35', 'hex10', 'spacer'], keep='first')
    #rev_TSS_df = rev_TSS_df.drop_duplicates(subset=['hex35', 'hex10', 'spacer', 'ITR'], keep='last')
    rev_TSS_df = rev_TSS_df.drop_duplicates(subset=['hex35', 'hex10', 'spacer'], keep='first')

    fwd_TSS_df = fwd_TSS_df.sort_values(by = 'Tx_rate', ascending = False)
    rev_TSS_df = rev_TSS_df.sort_values(by = 'Tx_rate', ascending = False)

    fwd_TSS_df['AA_Promoter_35'] = fwd_TSS_df['hex35'].apply(lambda x: str(Seq(x).translate()))
    fwd_TSS_df = fwd_TSS_df[fwd_TSS_df['AA_Promoter_35'].str.contains('*', regex = False) == False]

    fwd_TSS_df['AA_Promoter_10'] = fwd_TSS_df['hex10'].apply(lambda x: str(Seq(x).translate()))
    fwd_TSS_df = fwd_TSS_df[fwd_TSS_df['AA_Promoter_10'].str.contains('*', regex = False) == False]

    rev_TSS_df['AA_Promoter_35'] = rev_TSS_df['hex35'].apply(lambda x: str(Seq(x).translate()))
    rev_TSS_df = rev_TSS_df[rev_TSS_df['AA_Promoter_35'].str.contains('*', regex = False) == False]

    rev_TSS_df['AA_Promoter_10'] = rev_TSS_df['hex10'].apply(lambda x: str(Seq(x).translate()))
    rev_TSS_df = rev_TSS_df[rev_TSS_df['AA_Promoter_10'].str.contains('*', regex = False) == False]

    #TOP 5 Tx_rate records / Top1
    #minimise max values
    #maximise min values
    min_fwd_TSS_df = fwd_TSS_df.head(1)
    def_fwd_min_tx_rate = min_fwd_TSS_df['Tx_rate'].values[0]

    max_fwd_TSS_df = fwd_TSS_df.tail(1)
    def_fwd_max_tx_rate = max_fwd_TSS_df['Tx_rate'].values[0]


    min_rev_TSS_df = rev_TSS_df.head(1)
    def_rev_min_tx_rate = min_rev_TSS_df['Tx_rate'].values[0]

    max_rev_TSS_df = rev_TSS_df.tail(1)
    def_rev_max_tx_rate = max_rev_TSS_df['Tx_rate'].values[0]

    max_min_tx_rate_df = {"sequence": sequence, "sequence_compl": str(Seq(sequence).reverse_complement()), "max_fwd": def_fwd_max_tx_rate, "max_rev": def_rev_max_tx_rate, "min_fwd": def_fwd_min_tx_rate, "min_rev": def_rev_min_tx_rate, "prom_opt": get_promoter_opt()}

    #processes 10 records to maximise and 10 to minimise
    print("Minimising transcription rate on the forward strand..")
    fwd_res_df_min = pssm_util.process_df_promoters(fwd_TSS_df.head(10), 'fwd', 'min', max_min_tx_rate_df)
    ##fwd_res_df_min = pssm_util.process_df_promoters(fwd_TSS_df, 'fwd', 'min', max_min_tx_rate_df)

    print("Minimising transcription rate on the reverse strand..")
    rev_res_df_min = pssm_util.process_df_promoters(rev_TSS_df.head(10), 'rev', 'min', max_min_tx_rate_df)
    ##rev_res_df_min = pssm_util.process_df_promoters(rev_TSS_df, 'rev', 'min', max_min_tx_rate_df)

    print("Maximising transcription rate on the forward strand..")
    fwd_res_df_max = pssm_util.process_df_promoters(fwd_TSS_df.tail(10), 'fwd', 'max', max_min_tx_rate_df)
    ##fwd_res_df_max = pssm_util.process_df_promoters(fwd_TSS_df, 'fwd', 'max', max_min_tx_rate_df)

    print("Maximising transcription rate on the reverse strand..")
    rev_res_df_max = pssm_util.process_df_promoters(rev_TSS_df.tail(10), 'rev', 'max', max_min_tx_rate_df)
    ##rev_res_df_max = pssm_util.process_df_promoters(rev_TSS_df, 'rev', 'max', max_min_tx_rate_df)


    #res_final_df_max = pd.concat([max_fwd_TSS_df, max_rev_TSS_df], ignore_index=True, sort=False)
    res_final_df_max = pd.concat([fwd_res_df_max, rev_res_df_max], ignore_index=True, sort=False)

    #res_final_df_min = pd.concat([min_fwd_TSS_df, min_rev_TSS_df], ignore_index=True, sort=False)
    res_final_df_min = pd.concat([fwd_res_df_min, rev_res_df_min], ignore_index=True, sort=False)

    #res_final_df = res_final_df.drop_duplicates()
    res_final_df_max = res_final_df_max.drop_duplicates(
        subset=['hex35', 'hex10', 'Tx_rate', 'UP', 'ITR', 'Type', 'direction'],
        keep='last').reset_index(drop=True)


    res_final_df_min = res_final_df_min.drop_duplicates(
        subset=['hex35', 'hex10', 'Tx_rate', 'UP', 'ITR', 'Type', 'direction'],
        keep='last').reset_index(drop=True)

    res_final_df_max['hex35_aa'] = res_final_df_max['hex35_9nt'].apply(lambda x: str(Seq(x).translate()))
    res_final_df_min['hex35_aa'] = res_final_df_min['hex35_9nt'].apply(lambda x: str(Seq(x).translate()))
    res_final_df_max['hex10_aa'] = res_final_df_max['hex10_9nt'].apply(lambda x: str(Seq(x).translate()))
    res_final_df_min['hex10_aa'] = res_final_df_min['hex10_9nt'].apply(lambda x: str(Seq(x).translate()))

    #+PSSM values
    res_final_df_max['PSSM_hex35'] = res_final_df_max['hex35'].apply(lambda x: pssm_util.calc_PSSM(x, 0, '35'))
    res_final_df_max['PSSM_hex10'] = res_final_df_max['hex10'].apply(lambda x: pssm_util.calc_PSSM(x, 0, '10'))
    res_final_df_min['PSSM_hex35'] = res_final_df_min['hex35'].apply(lambda x: pssm_util.calc_PSSM(x, 0, '35'))
    res_final_df_min['PSSM_hex10'] = res_final_df_min['hex10'].apply(lambda x: pssm_util.calc_PSSM(x, 0, '10'))

    #perm_prom_pssm_df['PSSM_Promoters_perm'] = perm_prom_pssm_df['Promoters_perm_nt'].apply(lambda x: calc_PSSM(x, type))

    #res_final_df_max.to_csv('SalisLogelPromoterCalculator_MAX_results.csv')
    #res_final_df_min.to_csv('SalisLogelPromoterCalculator_MIN_results.csv')

    print("The processed gene sequence is " + sequence)

    res_final_df_max = res_final_df_max.rename(columns={'hex35_9nt': 'hex35_sequence', 'hex10_9nt': 'hex10_sequence'})
    res_final_df_min = res_final_df_min.rename(columns={'hex35_9nt': 'hex35_sequence', 'hex10_9nt': 'hex10_sequence'})


    res_final_df_max_fwd_df = res_final_df_max.loc[res_final_df_max["direction"] == 'fwd']
    new_max_fwd_Tx_rate_df = res_final_df_max_fwd_df.sort_values(by='Tx_rate', ascending=False)
    new_min_max_fwd_Tx_rate = new_max_fwd_Tx_rate_df['Tx_rate'].tail(1).values[0]
    new_max_max_fwd_Tx_rate = new_max_fwd_Tx_rate_df['Tx_rate'].head(1).values[0]


    res_final_df_min_fwd_df = res_final_df_min.loc[res_final_df_min["direction"] == 'fwd']
    new_min_fwd_Tx_rate_df = res_final_df_min_fwd_df.sort_values(by='Tx_rate', ascending=False)
    new_max_min_fwd_Tx_rate = new_min_fwd_Tx_rate_df['Tx_rate'].head(1).values[0]
    new_min_min_fwd_Tx_rate = new_min_fwd_Tx_rate_df['Tx_rate'].tail(1).values[0]
    #new_min_fwd_Tx_rate = new_min_fwd_Tx_rate_df['Tx_rate'].head(1).values[0]


    res_final_df_max_rev_df = res_final_df_max.loc[res_final_df_max["direction"] == 'rev']
    new_max_rev_Tx_rate_df = res_final_df_max_rev_df.sort_values(by='Tx_rate', ascending=False)
    new_max_max_rev_Tx_rate = new_max_rev_Tx_rate_df['Tx_rate'].head(1).values[0]
    new_min_max_rev_Tx_rate = new_max_rev_Tx_rate_df['Tx_rate'].tail(1).values[0]

    res_final_df_min_rev_df = res_final_df_min.loc[res_final_df_min["direction"] == 'rev']
    new_min_rev_Tx_rate_df = res_final_df_min_rev_df.sort_values(by='Tx_rate', ascending=False)
    new_max_min_rev_Tx_rate = new_min_rev_Tx_rate_df['Tx_rate'].head(1).values[0]
    new_min_min_rev_Tx_rate = new_min_rev_Tx_rate_df['Tx_rate'].tail(1).values[0]



    #column_list =  ["Type", "TSS", "Tx_rate", "Tx_rate_FoldChange", "hex35", "PSSM_hex35", "AA_hex35", "hex10", "PSSM_hex10", "AA_hex10", "UP", "spacer", "disc", "ITR", "new_gene_sequence", "promoter_sequence", "dG_total", "dG_10", "dG_35", "dG_disc", "dG_ITR", "dG_ext10", "dG_spacer", "dG_UP", "dG_bind",  "UP_position", "hex35_position", "spacer_position", "hex10_position", "disc_position"]
    column_list =  ["Type", "TSS", "Tx_rate", "Tx_rate_FoldChange", "frame", "hex35", "PSSM_hex35", "hex35_sequence", "hex35_aa", "hex10", "PSSM_hex10", "hex10_sequence", "hex10_aa", "UP", "spacer", "disc", "ITR", "new_gene_sequence", "promoter_sequence", "dG_total", "dG_10", "dG_35", "dG_disc", "dG_ITR", "dG_ext10", "dG_spacer", "dG_UP", "dG_bind",  "UP_position", "hex35_position", "spacer_position", "hex10_position", "disc_position"]

    ##max_fwd_TSS_df = fwd_res_df_max.loc[fwd_res_df_max['Tx_rate'].astype(float)
    ##max_fwd_TSS_df = fwd_res_df_max.loc[fwd_res_df_max['Tx_rate'].astype(float) >= float(def_fwd_max_tx_rate)]

    print("\nThe maximum transcription rate for the sequence (forward) is " + str(new_max_min_fwd_Tx_rate))
    if len(res_final_df_min_fwd_df) > 1:
        min_fwd_output_file = OUTPUT_FILE_NAME + "_MIN_FWD_results.csv"
        #if float(new_min_fwd_Tx_rate) < float(def_fwd_min_tx_rate):
        if float(new_min_min_fwd_Tx_rate) < float(new_max_min_fwd_Tx_rate):
            print ("can be decreased up to " + str(new_min_min_fwd_Tx_rate))
            print(USE_PROMOTERS_OUTPUT + "(" + min_fwd_output_file + ")")
            #res_final_df_min_fwd_df['Tx_rate_FoldChange'] = res_final_df_min_fwd_df['Tx_rate'].copy()/def_fwd_min_tx_rate.astype(float)
            res_final_df_min_fwd_df = pssm_util.add_txrate_foldchange_col(res_final_df_min_fwd_df, 'min')
            ##res_final_df_min_fwd_df.to_csv(min_fwd_output_file, columns = column_list, float_format='%.2f')
            res_final_df_min_fwd_df.to_csv(min_fwd_output_file, columns = column_list)

            ##files.download(min_fwd_output_file)

    else:
        print(" cannot be further decreased")

    print("\nThe maximum transcription rate for the sequence (reverse) is " + str(new_max_min_rev_Tx_rate))
    if len(res_final_df_min_rev_df) > 1:
        min_rev_output_file = OUTPUT_FILE_NAME + "_MIN_REV_results.csv"
        #if float(new_min_rev_Tx_rate) < float(def_rev_min_tx_rate):
        if float(new_min_min_rev_Tx_rate) < float(new_max_min_rev_Tx_rate):
            print ("can be decreased up to " + str(new_min_min_rev_Tx_rate))
            print(USE_PROMOTERS_OUTPUT + "(" + min_rev_output_file + ")")
            res_final_df_min_rev_df = pssm_util.add_txrate_foldchange_col(res_final_df_min_rev_df, 'min')
            #res_final_df_min_rev_df.to_csv(min_rev_output_file, columns = column_list, float_format='%.2f')
            res_final_df_min_rev_df.to_csv(min_rev_output_file, columns = column_list)

            ##files.download(min_rev_output_file)

    else:
        print(" cannot be further decreased")

    print("\nThe minimum transcription rate for the sequence (forward) is " + str(new_min_max_fwd_Tx_rate))
    if len(res_final_df_max_fwd_df) > 1:
        max_fwd_output_file = OUTPUT_FILE_NAME + "_MAX_FWD_results.csv"
        #if float(new_max_fwd_Tx_rate) > float(def_fwd_max_tx_rate):
        if float(new_max_max_fwd_Tx_rate) > float(new_min_max_fwd_Tx_rate):
            print ("can be increased up to " + str(new_max_max_fwd_Tx_rate))
            print(USE_PROMOTERS_OUTPUT + "(" + max_fwd_output_file + ")")
            res_final_df_max_fwd_df = pssm_util.add_txrate_foldchange_col(res_final_df_max_fwd_df, 'max')

            #res_final_df_max_fwd_df.to_csv(max_fwd_output_file, columns = column_list, float_format='%.2f')
            res_final_df_max_fwd_df.to_csv(max_fwd_output_file, columns = column_list)
            ##files.download(max_fwd_output_file)
    else:
        print(" cannot be further increased")

    print("\nThe minimum transcription rate for the sequence (reverse) is " + str(new_min_max_rev_Tx_rate))
    if len(res_final_df_max_rev_df) > 1:
        max_rev_output_file = OUTPUT_FILE_NAME + "_MAX_REV_results.csv"
        #if float(new_max_rev_Tx_rate) > float(def_rev_max_tx_rate):
        if float(new_max_max_rev_Tx_rate) > float(new_min_max_rev_Tx_rate):
            print ("can be increased up to " + str(new_max_max_rev_Tx_rate))
            print(USE_PROMOTERS_OUTPUT + "(" + max_rev_output_file + ")")
            res_final_df_max_rev_df = pssm_util.add_txrate_foldchange_col(res_final_df_max_rev_df, 'max')

            #res_final_df_max_rev_df.to_csv(max_rev_output_file, columns = column_list, float_format='%.2f')
            res_final_df_max_rev_df.to_csv(max_rev_output_file, columns = column_list)
            ##files.download(max_rev_output_file)

    else:
        print(" cannot be further increased")

    print("\n")

    print("\n")


    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))