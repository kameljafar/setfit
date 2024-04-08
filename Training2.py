import datasets
import os
import numpy as np
import pandas as pd
from setfit import SetFitModel
from mytrainer import MySetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset
import evaluate
from sklearn.metrics import hamming_loss,classification_report
from models.test.labels_information import TRAINING_LABELS
from config import SAVE_DIRECTORY,TRAIN_FILE, TEST_FILE
LABELS = TRAINING_LABELS
save_directory = SAVE_DIRECTORY
multilabel_f1_metric = evaluate.load("f1", "multilabel")
multilabel_accuracy_metric = evaluate.load("accuracy", "multilabel")
train_data = TRAIN_FILE
test_data = TEST_FILE


training = pd.read_csv('./data/train2.csv')
testing = pd.read_csv('./data/test2.csv')





training = training.sample(n=4)
testing = training.sample(n=2)

train_dataframe = training[['Unnamed: 0', 'text','cates']]
test_dataframe = testing[['Unnamed: 0', 'text','cates']]


train_dataframe = train_dataframe.rename(columns={'Unnamed: 0': 'index'})
test_dataframe = test_dataframe.rename(columns={'Unnamed: 0': 'index'})


train_dataframe['new_cates'] = train_dataframe['cates'].apply(lambda x : str(x).split(','))
test_dataframe['new_cates'] = test_dataframe['cates'].apply(lambda x : str(x).split(','))

def hotvec_multilabel(true_df):
    data = {}

    for i in range(len(true_df)):
        true_row = true_df.iloc[i]
        key = true_row['index']
        data[key] = set()
        if not pd.isna(true_row['label']):
            for l in true_row['label'].split(','):
                data[key].add(LABELS.index(l))
    y_hotvec = []
    for k, (true) in data.items():
        t = [0] * len(LABELS)
        
        for i in true:
            t[i] = 1
        y_hotvec.append(t)
    y_hotvec = np.array(y_hotvec)
    return y_hotvec

model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest")

# train_Ques_df = pd.DataFrame({'index':train_dataframe['index'],'text': train_dataframe['question'], 'label': train_dataframe['cates']})
# train_Title_df = pd.DataFrame({'index':train_dataframe['index'],'text': train_dataframe['title'], 'label': train_dataframe['cates']})
new_train_df = pd.DataFrame({'index':train_dataframe['index'],'text': train_dataframe['text'], 'label': train_dataframe['cates']})


# test_Ques_df = pd.DataFrame({'index':test_dataframe['index'],'text': test_dataframe['question'], 'label': test_dataframe['cates']})
# test_Title_df = pd.DataFrame({'index':test_dataframe['index'],'text': test_dataframe['title'], 'label': test_dataframe['cates']})

new_test_df = pd.DataFrame({'index':test_dataframe['index'],'text': test_dataframe['text'], 'label': test_dataframe['cates']})


# new_train_df = pd.concat([train_Ques_df, train_Title_df], axis=0)
# new_train_df.reset_index(drop=True, inplace=True)
# k = 1 
# for i in range (len(new_train_df)):
#     new_train_df['index'][i] =new_train_df['index'][i]+k
#     k+=1

# new_test_df = pd.concat([test_Ques_df, test_Title_df], axis=0)
# new_test_df.reset_index(drop=True, inplace=True)
# t = 1 
# for j in range (len(new_test_df)):
#     new_test_df['index'][j] =new_test_df['index'][j]+t
#     t+=1

label_train_array = np.array(hotvec_multilabel(new_train_df))
label_test_array = np.array(hotvec_multilabel(new_test_df))

new_train_label_array = np.resize(label_train_array, (len(new_train_df), len(LABELS)))
new_test_label_array = np.resize(label_test_array, (len(new_test_df) , len(LABELS)))

train_dataset = Dataset.from_dict({"index":new_train_df['index'],"text": new_train_df['text'], "label": new_train_label_array})
eval_dataset = Dataset.from_dict({'index':new_test_df['index'],"text": new_test_df['text'], "label": new_test_label_array})


def compute_metrics(y_pred, y_test):
    global report
    report = classification_report(y_pred, y_test)
    ham = hamming_loss(y_pred, y_test)
    return {
        "accuracy": multilabel_accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"] ,
        "f1": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="micro")["f1"] ,
        "f1_macro": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="macro")["f1"] ,
        "f1_weighted": multilabel_f1_metric.compute(predictions=y_pred, references=y_test, average="weighted")["f1"] ,
        "hamming_score": ham ,
    }


trainer = MySetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric=compute_metrics,
    batch_size=4,
    num_epochs=1,
    num_iterations=1,
    column_mapping={"text": "text", "label": "label"},
)


trainer.train()


model._save_pretrained(save_directory)

metrics = trainer.evaluate()

metrics_file = os.path.join(save_directory, "training_metrics.txt")
report_path = os.path.join(save_directory, "report.txt")
with open(metrics_file, 'w') as f:
    f.write(str(metrics))
with open(report_path, 'w') as f:
    f.write(str(report))

