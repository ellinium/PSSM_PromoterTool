**PSSM Promoter Tool**

The tool applies CORPSE (Codon Restrained Promoter Silencing) method and inverted CORPSE (iCORPSE) to the provided gene sequence.

-35 and -10 promoters along with the additional non-canonical sequence motifs are predicted based on the Salis Lab Promoter Calculator (https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator).
Position-specific scoring matrix (PSSM) is applied to all the synonymous codon variants of the promoters associated with the lowest and highest transcription rates in order to maximally decrease (CORPSE) or increase the transcription rate (iCORPSE).
The output CSV file/files contain synonymous codon promoters and sequence motifs for the minimal and maximal transcriptional rates along with the non-canonical sequence motifs for forward and reverse strands.

INSTALLATION:

USAGE:

The tool requires a text or fasta file with a nucleotide sequence of a gene to process.
```
python3 PSSMPromoterCalculator.py <file_name>
```

Depending on the result, up to four output CSV files can be generated:
1) PSSMPromoterCalculator_MAX_FWD_results.csv - contains promoters to minimise transcription rate (forward strand)
2) PSSMPromoterCalculator_MAX_REV_results.csv - contains promoters to minimise transcription rate (reverse strand)
3) PSSMPromoterCalculator_MIN_FWD_results.csv - contains promoters to maximise transcription rate (forward strand)
4) PSSMPromoterCalculator_MIN_REV_results.csv - contains promoters to maximise transcription rate (reverse strand).

The output file fields:

References:

1. Logel DY, Trofimova E, Jaschke PR. Codon-Restrained Method for Both Eliminating and Creating Intragenic Bacterial Promoters. ACS Synth Biol. 2022 Jan 19;acssynbio.1c00359. Available from https://pubs.acs.org/doi/10.1021/acssynbio.1c00359. doi: 10.1021/acssynbio.1c00359
2. LaFleur TL, Hossain A, Salis HM. Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria. Nat Commun. 2022 Sep 2;13(1):5159. Available from https://www.nature.com/articles/s41467-022-32829-5. doi: 10.1038/s41467-022-32829-5