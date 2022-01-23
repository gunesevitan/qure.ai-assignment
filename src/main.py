import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

import settings
import preprocessing
from datasets import NewsDataset
from models import DistilBERT
import trainer


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_news():

    labels = [
        'art',
        'economy',
        'education',
        'industry',
        'international',
        'national',
        'science & technology',
        'sports'
    ]

    # News content is scraped from the given URL and passed to the dataset
    news_url = request.form['news_url']
    news_response = BeautifulSoup(requests.get(news_url).text, 'html.parser')
    news_content = '\n'.join([p.text.strip() for p in news_response.select('div.article div div p')])
    dataset = NewsDataset(
        texts=[news_content],
        labels=None,
        tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        max_seq_len=256
    )

    # Load model and predict news content with it
    device = torch.device('cuda')
    model = DistilBERT(**config['model_parameters']['model_class_parameters'])
    model_path = f'{config["model_parameters"]["model_path"]}/{config["model_parameters"]["model_filename"]}.pt'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(
            input_ids=torch.unsqueeze(dataset[0]['input_ids'], dim=0).to(device),
            attention_mask=torch.unsqueeze(dataset[0]['attention_mask'], dim=0).to(device)
        )
    prediction = int(np.argmax(output.detach().cpu().numpy(), axis=1)[0])

    return render_template('predict.html', news_content=news_content, prediction=labels[prediction])


if __name__ == '__main__':

    # config_path and mode is required to start main
    # config consist of model and training parameters
    # mode can be selected as train, inference and deploy
    # - train: starts training with specified config
    # - inference: predict and evaluate trained model with specified config
    # - deploy: load model and serve web app
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
    elif args.mode == 'deploy':
        app.run(host='127.0.0.1', port=8080, debug=True)
