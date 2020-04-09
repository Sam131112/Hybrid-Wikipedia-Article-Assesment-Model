from handcraftedFeatures import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    le = LabelEncoder()
    df = pd.read_csv("wiki_pages_clean.csv")
    le.fit(df["Label"])
    df["label"] = le.transform(df["Label"])
    df = df[["Clean_Text","label"]]
    df = df.join(df["Clean_Text"].apply(readibility_feats))
    df["Article_to_bytes"] = df["Clean_Text"].apply(article_to_bytes)
    df["References"] = df["Clean_Text"].apply(references)
    df["In_links"] = df["Clean_Text"].apply(in_links)
    df["Num_templates"] = df["Clean_Text"].apply(num_templates)
    df["Num_categories"] = df["Clean_Text"].apply(num_categories)
    df["Img_by_article_len"] = df["Clean_Text"].apply(image_by_article_len)
    df["Information_to_noise"] = df["Clean_Text"].apply(information_to_noise)
    df["infobox"] = df["Clean_Text"].apply(infobox)
    df["Level2head"] = df["Clean_Text"].apply(level2head)
    df["Level3head"] = df["Clean_Text"].apply(level3head)
    cols = ["flesch_reading_ease","smog_index","flesch_kincaid_grade","coleman_liau_index","automated_readability_index","dale_chall_readability_score","difficult_words","linsear_write_formula","gunning_fog","Article_to_bytes","References","In_links","Num_templates","Img_by_article_len","Information_to_noise","Level2head","Level3head"]
    for col in cols:
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    
    train , test = train_test_split(df,test_size=0.25,shuffle=True,random_state=42,stratify=df["label"])
    train , validation = train_test_split(train,test_size=0.25,shuffle=True,random_state=42,stratify=train["label"])
    train.to_csv("Train.csv",index=False,header=True)
    test.to_csv("Test.csv",index=False,header=True)
    validation.to_csv("Validation.csv",index=False,header=True)
    df.to_csv("Full_Data.csv",index=False,header=True)


def meta(x):

    all_feats = ["infobox","flesch_reading_ease","smog_index","flesch_kincaid_grade","coleman_liau_index","automated_readability_index","dale_chall_readability_score","difficult_words","linsear_write_formula","gunning_fog","Article_to_bytes","References","In_links","Num_templates","Img_by_article_len","Information_to_noise","Level2head","Level3head"]
    txt = ""
    for row in all_feats:
        txt = txt + " " + str(x[row])
    return txt



def main1():

    all_feats = ["infobox","flesch_reading_ease","smog_index","flesch_kincaid_grade","coleman_liau_index","automated_readability_index","dale_chall_readability_score","difficult_words","linsear_write_formula","gunning_fog","Article_to_bytes","References","In_links","Num_templates","Img_by_article_len","Information_to_noise","Level2head","Level3head","Num_categories"]
    print("Feature Count ",len(all_feats))
    df = pd.read_csv("Full_Data.csv")
    df["Feats"] = df.apply(meta,axis=1)
    for feat in all_feats:
        df = df.drop(feat,axis=1)
    train , test = train_test_split(df,test_size=0.25,shuffle=True,random_state=42,stratify=df["label"])
    train , validation = train_test_split(train,test_size=0.25,shuffle=True,random_state=42,stratify=train["label"])
    train.to_csv("Train.csv",index=False,header=True)
    test.to_csv("Test.csv",index=False,header=True)
    validation.to_csv("Validation.csv",index=False,header=True)
    print("New Columns ",df.columns)
    df.to_csv("Data_with_Feats.csv",index=False,header=True)


if __name__ == "__main__":
    main1()

