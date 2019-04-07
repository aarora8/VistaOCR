import csv
import json

language = "Farsi"
slam_datafile = '/exp/scale18/ocr/data/SLAM_2.0/Datatracker/SLAM_Datatracker.csv'
id_conversion = '/exp/scale18/ocr/data/SLAM_2.0/FINAL_SLAM_sanitized/{0}/{0}_converted_list.txt'.format(language)

id_mapping = dict()
with open(id_conversion, 'r') as fh:
    for line in fh:
        id_orig, id_slam = line.strip().split()

        # Strip extension
        id_orig = id_orig[:id_orig.rfind('.')]
        id_slam = id_slam.replace(".png","")

        id_mapping[id_orig] = id_slam

breakdown = {}
with open(slam_datafile, 'r') as fh:
    fh.readline()
    csv_reader = csv.reader(fh)
    for fields in csv_reader:
        script,doc_language,file_name,file_ext,doc_class,secondary_scripts,secondary_languages,general_notes,att_tables,att_multicol,att_fielded,att_multiscript,att_textwithimages,att_technical,att_overlaid,att_artifacts,att_font,full_file_name,current_stage,xml,pdf_original = fields

        if not file_name in id_mapping:
            if doc_language == language:
                print(file_name)
            continue

        if not doc_class in breakdown:
            breakdown[doc_class] = set()

        slam_id = id_mapping[file_name].replace(".png","")
        breakdown[doc_class].add(slam_id)


for key in breakdown:
    print("size of %s = %d" % (key, len(breakdown[key])))

print(breakdown['overlaid_text'])
#print(breakdown['short_misc'])
