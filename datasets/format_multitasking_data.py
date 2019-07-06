def main(filename):
    import pandas as pd

    df = pd.read_csv('multitasking.csv')
    df = df.sort_values(by=['name', 'id']).reset_index(drop=True)
    df['class'] = pd.factorize(df['name'])[0]
    df.to_csv(filename, index=False)
    print('final file saved to', filename)
