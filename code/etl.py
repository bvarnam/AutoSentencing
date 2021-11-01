import pandas as pd
import numpy as np
import warnings

numeric_cols = ['AGE', 'AMTTOTAL', 'DRUGMIN', 'METHMIN', 'MWEIGHT', 'NODRUG',
                'NUMDEPEN', 'PROBATN', 'REGEXMIN', 'RELMIN', 'SENSPCAP', 'SENSPLT0',
                'SMAX1', 'SMIN1', 'STATMAX', 'STATMIN', 'SUPERMAX', 'SUPERMIN', 'SUPREL',
                'TIMESERVC', 'TOTCHPTS', 'TOTREST', 'TOTUNIT','MWGT1', 'WGT1', 'XFOLSOR',
                'XMAXSOR', 'XMINSOR']

categorical_cols = ['ACCGDLN', 'ALTDUM', 'CASETYPE', 'CITWHERE', 'COMBDRG2', 'CRIMHIST',
                       'DISPOSIT', 'DISTRICT', 'DSPLEA', 'EDUCATN', 'INTDUM', 'MONRACE',
                       'MONSEX', 'NEWCIT', 'NEWCNVTN', 'NEWEDUC', 'NEWRACE', 'OFFGUIDE', 
                       'PRISDUM', 'PROBDUM', 'QUARTER', 'REAS1', 'REAS2', 'REAS3', 'RESTDET1',
                       'RESTDUM', 'SAFE', 'SAFETY', 'SENTIMP', 'SOURCES', 'SUPRDUM', 'TYPEMONY',
                       'TYPEOTHS', 'UNIT1', 'XCRHISSR', 'SENTRNGE']
    
features = numeric_cols + categorical_cols

def mm_impute(df, mode_cols, mean_cols):
    for col in df.columns:
        if col in mode_cols:
            mode = df[col].value_counts().index[0]
            if not mode:
                mode = df[col].value_counts().index[1]
            df[col] = df[col].fillna(value = mode)
        elif col in mean_cols:
            df[col] = df[col].fillna(value = df[col].mean())
    return df

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    input_path = input("Path of file to be cleaned: ")
    output_path = input('Filepath of the cleaned file: ')
    df = pd.read_csv(input_path)
    print('Finished reading in the file!')
    df_new = pd.DataFrame()
    for col in features:
        try:
            df_new[col] = df[col]
        except:
            pass
    df = df_new[(df_new['OFFGUIDE'] == 9) | (df_new['OFFGUIDE'] == 10)]
    df = mm_impute(df, categorical_cols, numeric_cols)
    df = df[df.columns[df.isnull().sum() < 10000]]
    df.columns = [col.lower() for col in df.columns]
    df.to_csv(output_path, index = False)
    print('Your cleaned file can be found at', output_path)

