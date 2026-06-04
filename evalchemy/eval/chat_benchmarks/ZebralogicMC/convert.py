import pandas as pd
from datasets import load_dataset


def reformat_zebralogic_mc(df):
    def format_choices(choices):
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        return '\n'.join([f"({labels[i]}) {c}" for i, c in enumerate(choices)])

    def format_answer(answer, choices):
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        try:
            idx = list(choices).index(answer)
            return f"({labels[idx]})"
        except ValueError:
            return answer

    df['input'] = (
        df['puzzle'] +
        "\nQuestion: " + df['question'] +
        "\nChoices:\n" + df['choices'].apply(format_choices) +  
        "\n\nAnswer by providing only the correct option letter (e.g., A, B, C, D, E, etc.)."
    )
    df['target'] = df.apply(lambda row: format_answer(row['answer'], row['choices']), axis=1)

    return df[['input', 'target']]


ds = load_dataset('WildEval/ZebraLogic', 'mc_mode', split='test')
df = ds.to_pandas()
reformatted_df = reformat_zebralogic_mc(df)
print(f"Total examples: {len(reformatted_df)}")
print(reformatted_df.head())
reformatted_df.to_json('zebralogic_mc_reformatted.json', orient='records', lines=True)