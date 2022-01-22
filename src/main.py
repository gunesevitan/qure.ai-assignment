import os
import argparse
import yaml
import pandas as pd


import settings
import preprocessing
import trainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    df = pd.read_csv(settings.DATA / 'news.csv')
    preprocessor = preprocessing.PreprocessingPipeline(df=df)
    df = preprocessor.transform()

    classification_trainer = trainer.ClassificationTrainer(
        model_parameters=config['model_parameters'],
        training_parameters=config['training_parameters']
    )

    if args.mode == 'train':
        classification_trainer.train_and_validate(df)
    elif args.mode == 'inference':
        classification_trainer.inference(df)
