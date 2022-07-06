# standard libraries
import csv

# 3rd party libraries
import json
import codecs
from numpy import inner
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame


def prep_step_a(df_input):
    w = 0
    x = 0
    y = 0
    list_sentence_id = []
    list_entity_id = []
    list_offset_start = []
    list_offset_end = []
    df_output = df_input.copy()

    for i in tqdm(df_output.itertuples()):
        if pd.isna(i.token) != True and pd.isna(i.ne) != True:
            list_sentence_id.append(w)
            if i.ne != "O":
                if i.Index != 0:
                    y += 1
                    list_entity_id.append(x)
                    list_offset_start.append(y)
                    list_offset_end.append(y+len(i.token))
                    y += len(i.token)
                else:
                    list_entity_id.append(x)
                    list_offset_start.append(y)
                    list_offset_end.append(y+len(i.token))
                    y += len(i.token)
                x += 1
            else:
                if i.Index != 0:
                    y += 1
                    list_entity_id.append("-")
                    list_offset_start.append("-")
                    list_offset_end.append("-")
                    y += len(i.token)
                else:
                    list_entity_id.append("-")
                    list_offset_start.append("-")
                    list_offset_end.append("-")
                    y += len(i.token)
        else:
            list_sentence_id.append("-")
            list_entity_id.append("-")
            list_offset_start.append("-")
            list_offset_end.append("-")
            w += 1
            x = 0
            y = 0

    # seragamkan entity id yang bertetangga
    for idx, x in enumerate(list_entity_id):
        if x == "-":
            continue
        else:
            if type(list_entity_id[idx-1]) == int:
                if int(list_entity_id[idx])-int(list_entity_id[idx-1]) == 1:
                    list_entity_id[idx] = list_entity_id[idx-1]

    df_output["sentence_id"] = list_sentence_id
    df_output["entity_id"] = list_entity_id
    df_output["offset_start"] = list_offset_start
    df_output["offset_end"] = list_offset_end

    # print(list_offset_start)
    return df_output


def prep_steb_b(df_input, start_counter):
    # final
    list_singgalang = []
    counter = start_counter

    # temporary
    id_tmp_sent = 0
    list_tmp_token = []
    list_tmp_ne = []

    for i in tqdm(df_input.itertuples()):
        if pd.isna(i.token) != True and pd.isna(i.ne) != True:
            id_tmp_sent = i.sentence_id
            list_tmp_token.append(i.token)
        else:
            df_tmp = df_input[df_input["sentence_id"] == id_tmp_sent]
            df_tmp2 = df_tmp[~df_tmp["entity_id"].isin(["-"])]
            a = df_tmp2.groupby(["entity_id", "ne"], as_index=False)[
                "token"].agg(lambda x: list(x))
            b = df_tmp2.groupby(["entity_id", "ne"], as_index=False)[
                "offset_start"].agg(lambda x: list(x))
            c = df_tmp2.groupby(["entity_id", "ne"], as_index=False)[
                "offset_end"].agg(lambda x: list(x))

            if(a.empty == False):
                d = a.merge(b.merge(c, how='inner', on='entity_id'),
                            how='inner', on='entity_id')
                for i in range(len(d)):
                    #                     print(f'{counter}')
                    d.entity_id[i] = str(counter)
                    counter = counter + 1

            for j in d.itertuples():
                list_tmp_ne.append({
                    "entity_id": str(j.entity_id),
                    "text": " ".join(j.token),
                    "label": j.ne,
                    "start_offset": j.offset_start[0],
                    "end_offset": j.offset_end[-1]
                })
            # print(list_tmp_ne)

            list_singgalang.append({
                "doc_id": "Singgalang-"+str(id_tmp_sent),
                "doc_text": " ".join(list_tmp_token),
                "entities": list_tmp_ne
            })
            id_tmp_sent = 0
            list_tmp_token = []
            list_tmp_ne = []
    return list_singgalang


def main():
    counter = 1
    file_path = "SINGGALANG.tsv"
    df_raw = pd.read_csv(file_path,
                         sep="\t",
                         names=["token", "ne"],
                         skip_blank_lines=False,
                         quoting=csv.QUOTE_NONE,
                         encoding="utf-8")
    # df_raw = df_raw.loc[:576]
    # print(df_raw)
    # df_raw = df_raw.loc[:len(df_raw)-1]
    print("Proses restrukturisasi dataframe sedang dilakukan ...")
    df_res = prep_step_a(df_raw)
    print("Proses restrukturisasi dataframe selesai.")

    print("Proses finalisasi struktur set data sedang dilakukan ...")
    list_res = prep_steb_b(df_res, counter)
    print("Proses finalisasi struktur set data selesai.")
    # print(list_res)
    with open("singgalang.json", "wb") as f:
        json.dump(list_res,
                  codecs.getwriter("utf-8")(f),
                  ensure_ascii=False)

    print("\n")
    print("Hasil konversi \"singgalang.tsv\" ke \"singgalang.json\" selesai.")


if __name__ == "__main__":
    main()
