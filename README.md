# Evaluation of Yelp User Review Effectiveness(evaYURE)

## Setup Environment
1. Download dataset
Download the [Yelp dataset](https://www.yelp.com/dataset) and save at `./data`.
2. Setup virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Use?

1. Train NLP
```
python -m mlp nlp_train
```

2. Test NLP
```
python -m mlp nlp_test
```

3. Cache NLP Test Results
```
python -m mlp nlp_cache
```

4. Perform K-means
```
python -m mlp kmeans_learn
```

