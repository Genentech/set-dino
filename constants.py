from enum import Enum


class Column(Enum):
    plate = 'plate'
    well = 'well'
    tile = 'tile'
    sgRNA = 'sgRNA'
    gene = 'gene_symbol'


def build_train_arr():
    well_list = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    plate_list = ['20200202_6W-LaC024A', '20200202_6W-LaC024E', '20200206_6W-LaC025A',
                  '20200202_6W-LaC024D', '20200202_6W-LaC024F', '20200206_6W-LaC025B']
    train_arr = [(plate, well) for plate in plate_list for well in well_list]
    return train_arr


def build_val_arr():
    well_list = ['A1']
    plate_list = ['20200202_6W-LaC024A', '20200202_6W-LaC024E', '20200206_6W-LaC025A',
                  '20200202_6W-LaC024D', '20200202_6W-LaC024F', '20200206_6W-LaC025B']
    val_arr = [(plate, well) for plate in plate_list for well in well_list]
    return val_arr

def build_test_arr():
    well_list = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    plate_list = ['20200202_6W-LaC024B', '20200202_6W-LaC024C']
    test_arr = [(plate, well) for plate in plate_list for well in well_list]
    return test_arr


TRAIN = build_train_arr()
VAL = build_val_arr()
TEST = build_test_arr()

NTC = 'nontargeting'

"""
Selected genes with strong phenotype based on CellProfiler features.
Note this gene subset was only used for validation steps to reduce the computation burden. In final evaluation, 
we used all genetic perturbations.
Set GENES_FOR_VAL to None to run validation on all genes in the validation set 
"""
GENES_FOR_VAL = ['AAAS', 'AAMP', 'AARS1', 'AATF', 'ABCB7', 'ABCE1', 'ABCF1', 'ABHD17A', 'ACACA', 'ACBD4', 'ACD',
                 'ACIN1', 'ACTB', 'ACTG1', 'ACTL6A', 'ACTR2', 'ACTR8', 'ADNP', 'ADRM1', 'ADSS2', 'AFG3L2', 'AGAP4',
                 'AGAP6', 'AGBL4', 'AGO1', 'AGO2', 'AGPAT5', 'AHCTF1', 'AHCYL1', 'AIP', 'AK2', 'AKIRIN2', 'ALAS1',
                 'ALG11', 'ALS2CL', 'ALYREF', 'AMER2', 'AMY1A', 'AMY1B&AMY1C', 'ANAPC1', 'ANAPC10', 'ANAPC15',
                 'ANKHD1', 'ANKLE2', 'ANKRD17', 'ANKRD20A1', 'ANKRD52', 'ANLN', 'AP2M1', 'AQR', 'ARCN1', 'AREL1',
                 'ARF1', 'ARF4', 'ARHGEF35', 'ARHGEF4', 'ARID3A', 'ARIH1', 'ARMC6', 'ASPSCR1', 'ASS1', 'ATAD5',
                 'ATF7IP', 'ATL2', 'ATMIN', 'ATP13A1', 'ATP1A1', 'ATP2A2', 'ATP5F1B', 'ATP5MG', 'ATP5MGL', 'ATP5PB',
                 'ATP6V0B', 'ATP6V0C', 'ATP6V1F', 'BANP', 'BAP1', 'BCAS2', 'BCL9L', 'BCR', 'BDP1', 'BIRC6', 'BLOC1S3',
                 'BORA', 'BORCS7', 'BRD1', 'BRF2', 'BUB3', 'BUD13', 'BUD23', 'C12orf45', 'C16orf72', 'C19orf25',
                 'C1QTNF4', 'C7orf26', 'CA5A', 'CAB39', 'CAD', 'CALCB', 'CAMKK2', 'CAPS', 'CCDC144A', 'CCDC174',
                 'CCDC51', 'CCDC61', 'CCDC71L', 'CCDC84', 'CCNL1', 'CCP110', 'CCT2', 'CDC34', 'CDC5L', 'CDK5RAP2',
                 'CEBPZ', 'CERT1', 'CHAMP1', 'CHD8', 'CHGB', 'CHMP7', 'CHORDC1', 'CHP1', 'CHRND', 'CHRNG', 'CIAO2B',
                 'CIB2', 'CLDN6', 'CLPX', 'COASY', 'COL5A3', 'COX7B', 'CPSF4', 'CSDE1', 'CTSF', 'CUL1', 'CUL2', 'CUX1',
                 'CYB5R4', 'CYCS', 'CYS1', 'DCAF13', 'DDX1', 'DDX21', 'DHPS', 'DNAAF2', 'DNAJA2', 'DNAJB12', 'DNAJB6',
                 'DPM3', 'DTD2', 'DYNC1H1', 'EFNA3', 'EIF3A', 'EIF3H', 'EIF3L', 'ELP2', 'ELP6', 'ENPP7', 'ERBIN',
                 'EXOC8', 'FAM104B', 'FAM153B', 'FAM185A', 'FAM72A', 'FBXO42', 'FBXO5', 'FIBCD1', 'FOXN4', 'FRG1',
                 'FUT4', 'G6PD', 'GALK1', 'GNPNAT1', 'GOT2', 'GTF2B', 'H1-4', 'H2AB2', 'H2AX', 'HERC2', 'HIP1R',
                 'HNRNPD', 'HSD17B12', 'HYOU1', 'ICE1', 'IER3IP1', 'INO80C', 'INTS1', 'KHDC4', 'KIF22', 'KRAS',
                 'LRCH2', 'LRRC37A3', 'LSM3', 'MAD2L2', 'MAFK', 'MAU2', 'MCL1', 'MCM4', 'MCPH1', 'MRPS26', 'NAA38',
                 'NCR2', 'NDUFA1', 'NDUFC2', 'NKX6-1', 'NTF4', 'NUP155', 'OVCA2', 'PDCL', 'PF4', 'PNKP', 'POLD1',
                 'PSMD1', 'RLF', 'RXRB', 'SEPHS1', 'SMARCA5', 'SOD2', 'SOX10', 'SPATA31A6', 'SYF2', 'TPSG1', 'VPS37B',
                 'nontargeting']