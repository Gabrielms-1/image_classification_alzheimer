import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def count_images_in_directory(directory):
    class_counts = defaultdict(int)
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.basename(root)
                class_counts[class_name] += 1
    return class_counts

def main():
    base_dir = 'data/raw'
    sets = ['train', 'test', 'valid']
    
    set_counts = {set_name: defaultdict(int) for set_name in sets}
    
    for set_name in sets:
        set_dir = os.path.join(base_dir, set_name)
        counts = count_images_in_directory(set_dir)
        for class_name, count in counts.items():
            set_counts[set_name][class_name] = count
    
    df = pd.DataFrame(set_counts).fillna(0).astype(int)
    df.columns = ['Train', 'Test', 'Valid']
    
    print("\nClass distribution table:")
    print(df)

    plt.figure(figsize=(12, 6))
    
    df.T.plot(kind='barh', stacked=True, figsize=(12, 6))
    plt.title('Class distribution between splits (Absolute values)')
    plt.xlabel('Number of samples')
    plt.ylabel('Split')
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    
    df_normalized = df.T.div(df.T.sum(axis=1), axis=0) * 100
    df_normalized.plot(kind='bar', stacked=True, figsize=(12, 6))
    
    plt.title('Class distribution between splits (Percentage)', pad=20)
    plt.xlabel('Split')
    plt.ylabel('Proportion (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()