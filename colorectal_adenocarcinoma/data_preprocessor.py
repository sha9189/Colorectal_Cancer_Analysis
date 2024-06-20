import pandas as pd
import sys

def show_unique_vals_by_column(df:pd.DataFrame):
    """Function to show distinct values in each column of a pandas dataframe"""
    numeric_cols = []
    for column in sorted(df.columns):
        if df[column].dtype == object:  # Check if the column data type is object (typically used for text)
            unique_values = df[column].unique()
            print(f"Column '{column}' has {len(unique_values)} unique values: {sorted(unique_values)[:10]} {'...' if len(unique_values) > 10 else ''}")
        else:
            numeric_cols.append(column)
    print(f"Ignored these numeric columns: {numeric_cols}")


def genetic_ancestry_to_eur_ind(race:str):
    """Function that takes GENETIC_ANCESTRY_LABEL as input and returns 1 for European(White) and 0 for non-European"""
    if 'EUR' in race:
        return 1
    else:
        return 0
    

def cleanup_prior_dx(prior_dx):
    """Remove extra details and return just Y/N"""
    if prior_dx.upper().startswith("YES"):
        return "Y"
    elif prior_dx.upper().startswith("NO"): 
        return "N"
    else:
        print("Unexpected value for 'PRIOR_DX'...exiting")
        sys.exit(0)


def label_encode_m_stage(m_stage:str):
    """
    Label encode m-stage. Can handle only : [M0, M1, M1A, M1B, MX] 
    """
    m_stage = m_stage.upper()
    if m_stage == 'M0':
        return 0
    elif m_stage == 'MX':
        return 0.5
    elif m_stage == 'M1' or m_stage == 'M1A':
        return 1
    elif m_stage == 'M1B':
        return 1.5
    else:
        print("Unexpected value for M-Stage...exiting")
        sys.exit(0)


def label_encode_t_stage(t_stage:str):
    """
    Label encode T-stage. Can handle only : ['T1', 'T2', 'T3', 'T4', 'T4A', 'T4B', 'TIS'] 
    """
    t_stage = t_stage.upper()
    if t_stage == 'TIS':
        return 0
    elif t_stage == 'T1':
        return 1
    elif t_stage == 'T2':
        return 2
    elif t_stage == 'T3':
        return 3
    elif t_stage == 'T4' or t_stage == 'T4A':
        return 4
    elif t_stage == 'T4B':
        return 4.5
    else:
        print("Unexpected value for T-Stage...exiting")
        sys.exit(0)


def label_encode_n_stage(n_stage:str):
    """
    Label encode N-stage. Can handle only : ['N0', 'N1', 'N1A', 'N1B', 'N1C', 'N2', 'N2A', 'N2B'] 
    """
    n_stage = n_stage.upper()
    
    # if n_stage == 'N0':
    if n_stage == 'N0' or n_stage == "NX": #temp line
        return 0
    elif n_stage == 'N1' or n_stage == 'N1A':
        return 1
    elif n_stage == 'N1B':
        return 1.33
    elif n_stage == 'N1C':
        return 1.66
    elif n_stage == 'N2' or n_stage == 'N2A':
        return 2
    elif n_stage == 'N2B':
        return 2.5
    else:
        print("Unexpected value for N-Stage...exiting")
        sys.exit(0)


def preprocess_data(df1:pd.DataFrame):
    """
    Function to preprocess data so that 
    1. TNM Stage, Prior Diagnosis, Genetic Ancestry Label columns are cleaned
    2. All object type columns are converted to one hot encoding
    """
    columns = df1.columns

    # label encode TNM Stage
    if 'PATH_M_STAGE' in columns:
        df1["PATH_M_STAGE"] = df1["PATH_M_STAGE"].apply(label_encode_m_stage)
        print("Label encoded PATH_M_STAGE")

    if 'PATH_N_STAGE' in columns:
        df1["PATH_N_STAGE"] = df1["PATH_N_STAGE"].apply(label_encode_n_stage)
        print("Label encoded PATH_N_STAGE")

    if 'PATH_T_STAGE' in columns:
        df1["PATH_T_STAGE"] = df1["PATH_T_STAGE"].apply(label_encode_t_stage)
        print("Label encoded PATH_T_STAGE")

    # cleanup prior diagno
    if 'PRIOR_DX' in columns:
        df1["PRIOR_DX"] = df1["PRIOR_DX"].apply(cleanup_prior_dx)
        
    if 'GENETIC_ANCESTRY_LABEL' in columns:
        df1['GENETIC_ANCESTRY_LABEL'] = df1["GENETIC_ANCESTRY_LABEL"].apply(genetic_ancestry_to_eur_ind)
        
    # Apply one-hot encoding to object type columns
    object_columns = df1.select_dtypes(include=['object']).columns
    df_one_hot_encoded = pd.get_dummies(df1, columns=object_columns, drop_first=True, dtype='int64')

    return df_one_hot_encoded
