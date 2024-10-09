import pandas as pd 

def split_hex(hex_color):
    return hex_color[1:3], hex_color[3:5], hex_color[5:7]

def convert_hex_to_rgb(hex_color):
    r, g, b = split_hex(hex_color)
    return int(r, 16), int(g, 16), int(b, 16)

if __name__ == '__main__':
    df = pd.read_csv('data/rgb.txt', sep='\t', header=None)
    df = df.drop(columns=[2])
    df.columns = ['color', 'hex']
    df = df[df['color'].str.split(' ').str.len() == 1]
    df['r'], df['g'], df['b'] = zip(*df['hex'].map(convert_hex_to_rgb))
    df.to_csv('data/rgb.csv', index=False)
