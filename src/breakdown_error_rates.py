import csv
import json
import os
import sys


def main():
    train_data_source = "Yomdle"
    test_data_source = "Slam"
    test_on_full_slam = True

    if test_on_full_slam:
        extra = "_FULL"
    else:
        extra = ""

    for lang in ['Farsi', 'Tamil', 'Russian', 'Korean']:
        #sclite_file = '/home/hltcoe/srawls/expts/vista-ocr/decode-results/{0}{1}_{2}{1}{3}/hyp-words.txt.pra'.format(train_data_source, lang, test_data_source, extra)
        sclite_file = '/exp/detter/scale18/data/derived/SLAM_2.0/Farsi/transcribed/hyp.snor.pra'

        if lang != 'Farsi': continue

        if test_on_full_slam:
            json_file = '/exp/scale18/ocr/users/srawls/{0}_{1}/desc.json'.format(test_data_source.lower(), lang.lower())
        else:
            json_file = '/exp/scale18/ocr/users/srawls/{0}_{1}_traindevtest/desc.json'.format(test_data_source.lower(), lang.lower())

        print("WER for Scale %s" % lang)
        print_breakdown(sclite_file, json_file, lang)


def print_breakdown(sclite_file, json_file, lang):
    sclite_scores = parse_sclite_pra(sclite_file)
    #breakdown = breakdown_by_height(json_file)

    breakdown = breakdown_by_type(lang)

    # Figure out gloabl error rate
    nerr = 0
    ntotal_ref = 0
    breakdown_scores = {}
    for key in breakdown:
        breakdown_scores[key] = {'nerr': 0, 'ntotal_ref': 0, 'nlines': 0}

    for uttid, entry in sclite_scores.items():
        correct, substitutions, deletions, insertions = entry
        nerr += substitutions + deletions + insertions
        ntotal_ref += correct + substitutions + deletions

        pageid = uttid[ :uttid.rfind('_') ]
        #pageid = pageid[ :pageid.rfind('_') ]
        for key in breakdown:
            #if uttid in breakdown[key]:
            if pageid in breakdown[key]:
                breakdown_scores[key]['nlines'] += 1
                breakdown_scores[key]['nerr'] += substitutions + deletions + insertions
                breakdown_scores[key]['ntotal_ref'] += correct + substitutions + deletions

    print("Total Error rate = %0.2f%%" % (100*nerr/ntotal_ref))
    for key in sorted(breakdown_scores.keys()):
        if breakdown_scores[key]['ntotal_ref'] == 0:
            print("\t%s Error rate = N/A  (nlines = 0)" % key)
        else:
            print("\t%s Error rate = %0.2f%%  (nlines = %d, n_total_tokens = %d)" % (key, (100*breakdown_scores[key]['nerr']/breakdown_scores[key]['ntotal_ref']),breakdown_scores[key]['nlines'], breakdown_scores[key]['ntotal_ref']))


def parse_sclite_pra(f):
    data = {}
    with open(f, 'r') as fh:
        for line in fh:
            if line.startswith('id: '):
                uttid = line[4:]
                if uttid.startswith("("):
                    uttid = line[ line.find('(') + 1 : line.find(')') ]
                else:
                    uttid = uttid[:-3]

                uttid = uttid.lower()
            if line.startswith('Scores: '):
                #Scores: ( #C #S #D #I ) = ( 13 2 0 0 )
                if '=' in line:
                    correct, sub, deletions, insertions = line[ line.rfind('(')+1 : line.rfind(')')].split()
                else:
                    correct, sub, deletions, insertions = line[ line.find(')')+1 :].split()


                data[uttid] = ( int(correct), int(sub), int(deletions), int(insertions) )

    return data

def breakdown_by_type(lang):
    slam_datafile = '/exp/scale18/ocr/data/SLAM_2.0/Datatracker/SLAM_Datatracker.csv'
    id_conversion = '/exp/scale18/ocr/data/SLAM_2.0/FINAL_SLAM_sanitized/{0}/{0}_converted_list.txt'.format(lang)

    id_mapping = dict()
    with open(id_conversion, 'r') as fh:
        for line in fh:
            id_orig, id_slam = line.strip().split()

            # Strip extension
            id_orig = id_orig[:id_orig.rfind('.')]
            id_slam = id_slam.replace(".png","")

            id_slam = id_slam.lower()
            id_mapping[id_orig] = id_slam

    breakdown = {}
    with open(slam_datafile, 'r') as fh:
        fh.readline()
        csv_reader = csv.reader(fh)
        for fields in csv_reader:
            script,language,file_name,file_ext,doc_class,secondary_scripts,secondary_languages,general_notes,att_tables,att_multicol,att_fielded,att_multiscript,att_textwithimages,att_technical,att_overlaid,att_artifacts,att_font,full_file_name,current_stage,xml,pdf_original = fields

            if not file_name in id_mapping:
                continue

            if not doc_class in breakdown:
                breakdown[doc_class] = set()

            slam_id = id_mapping[file_name].replace(".png","")
            breakdown[doc_class].add(slam_id)


    if 'letter' in breakdown and 'letter/memo' in breakdown:
        # Merge these, not sure why they are different only in the Russian data
        for uttid in breakdown['letter']:
            breakdown['letter/memo'].add(uttid)
        del breakdown['letter']

    return breakdown



def breakdown_by_height(f):
    with open(f, 'r') as fh:
        data = json.load(fh)


    ids_0_15 = []
    ids_16_25 = []
    ids_26_35 = []
    ids_36_45 = []
    ids_46_up = []

    for entry in data['test']:
        uttid = entry['id'].lower()

        h = entry['height']
        if h <= 15:
            ids_0_15.append(uttid)
        elif h <= 25:
            ids_16_25.append(uttid)
        elif h <= 35:
            ids_26_35.append(uttid)
        elif h <= 45:
            ids_36_45.append(uttid)
        else:
            ids_46_up.append(uttid)
            
    breakdown = { 'lh 0-15': set(ids_0_15),
                  'lh 16-25': set(ids_16_25),
                  'lh 26-35': set(ids_26_35),
                  'lh 36-45': set(ids_36_45),
                  'lh 46-up': set(ids_46_up),
              }

    return breakdown



if __name__ == "__main__":
    main()
