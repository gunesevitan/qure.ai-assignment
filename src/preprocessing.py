import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class PreprocessingPipeline:

    def __init__(self, df):

        self.df = df.copy(deep=True)

    def get_splits(self):

        """
        Split dataset into training/validation/test sets
        """

        # Training/validation/test splits with stratified categories
        df_train, df_val_test = train_test_split(self.df, stratify=self.df['category'], test_size=0.3)
        df_val, df_test = train_test_split(df_val_test, stratify=df_val_test['category'], test_size=0.66)
        df_train['fold'] = 'train'
        df_val['fold'] = 'val'
        df_test['fold'] = 'test'
        self.df = pd.concat((df_train, df_val, df_test), axis=0, ignore_index=True)

    def group_titles(self):

        """
        Group similar categories into groups
        """

        self.df.loc[self.df['category'] == 'other states', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'cricket', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'kerala', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'tamil nadu', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'delhi', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'companies', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'andhra pradesh', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'football', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'cinema', 'category'] = 'art'
        self.df.loc[self.df['category'] == 'karnataka', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'races', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'other sports', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'business', 'category'] = 'economy'
        self.df.loc[self.df['category'] == 'chennai', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'athletics', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'bengaluru', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'music', 'category'] = 'art'
        self.df.loc[self.df['category'] == 'tennis', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'motorsport', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'madurai', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'mangaluru', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'sci-tech', 'category'] = 'science & technology'
        self.df.loc[self.df['category'] == 'states', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'health', 'category'] = 'science & technology'
        self.df.loc[self.df['category'] == 'hockey', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'dance', 'category'] = 'art'
        self.df.loc[self.df['category'] == 'research', 'category'] = 'science & technology'
        self.df.loc[self.df['category'] == 'kalpana sharma', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'yogacharini maitreyi', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'sevanti ninan', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'harsh mander', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'hindol sengupta', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'vasundhara chauhan', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'vijay nagaswami', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'suchitra behal', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'shyam', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'tiruchirapalli', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'siddharth varadarajan', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'm.v. ramakrishnan', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'vijayawada', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'visakhapatnam', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'hyderabad', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'thiruvananthapuram', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'thiruvananthapuram', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'sport', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'fitness', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'cities', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'hasan suroor', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'medicine', 'category'] = 'science & technology'
        self.df.loc[self.df['category'] == 'sainath', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'technology', 'category'] = 'science & technology'
        self.df.loc[self.df['category'] == 'science', 'category'] = 'science & technology'
        self.df.loc[self.df['category'] == 'theatre', 'category'] = 'art'
        self.df.loc[self.df['category'] == 'colleges', 'category'] = 'education'
        self.df.loc[self.df['category'] == 'education plus', 'category'] = 'education'
        self.df.loc[self.df['category'] == 'kochi', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'coimbatore', 'category'] = 'national'
        self.df.loc[self.df['category'] == 'motoring', 'category'] = 'sports'
        self.df.loc[self.df['category'] == 'bill kirkman', 'category'] = 'international'

        self.df.loc[self.df['category'] == 'schools', 'category'] = 'education'
        self.df.loc[self.df['category'] == 'arts', 'category'] = 'art'
        self.df.loc[self.df['category'] == 'money & careers', 'category'] = 'economy'
        self.df.loc[self.df['category'] == 'gadgets', 'category'] = 'science & technology'

        # Remove vague categories since they can be noisy
        other_categories = [
            'new articles', 'news', 'comment', 'letters', 'editorial',
            'lead', 'rx', 'interview', 'internet', 'magazine', 'markets',
            'environment', 'cartoon', 'agriculture', 'open page', 'books',
            'television', 'history & culture', 'faith', 'fashion', 'life & style',
            'travel', 'issues', 'society', 'readers\' editor', 'diet & nutrition', 'food',
            'young world', 'policy & issues', 'property plus', 'crafts', 'careers'
        ]

        self.df = self.df.loc[~self.df['category'].isin(other_categories), :].reset_index(drop=True)
        self.df['category_labels'] = LabelEncoder().fit_transform(self.df['category'])

    def transform(self):

        self.group_titles()
        self.get_splits()

        return self.df
