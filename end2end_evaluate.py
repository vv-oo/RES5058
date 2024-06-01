import pandas as pd

test_df = pd.read_csv('Dataset/Sentiment/cosmetic_Sentiment_test.csv')
absa_df = pd.read_csv('ABSA.csv')

# remove blanks
test_df['Text'] = test_df['Text'].apply(lambda x: x.replace(' ', ''))
absa_df['Text'] = absa_df['Text'].apply(lambda x: x.replace(' ', ''))

absa_df.loc[absa_df['Sentiment'] == 2, 'Sentiment'] = -1

# accuracy
result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Aspect', 'Sentiment'], how='outer', indicator=True)
both = result[result['_merge'] == 'both']
test_only = result[result['_merge'] == 'left_only']
absa_only = result[result['_merge'] == 'right_only']
print(len(both), len(test_df), len(both) / len(test_df))

# save compare result
result.to_csv('Results/compare_result.csv', encoding='UTF-8', index=False)