import json

#data_file = '/exp/scale18/ocr/users/srawls/yomdle_russian_traindevtest/desc.json'
data_file = '/exp/scale18/ocr/users/srawls/slam_russian_traindevtest/desc.json'
with open(data_file, 'r') as fh:
    data = json.load(fh)


ids_0_9 = []
ids_10_15 = []
ids_16_20 = []
ids_21_25 = []
ids_26_30 = []
ids_31_40 = []
ids_41_50 = []
ids_51_up = []

for entry in data['test']:
    uttid = entry['id']
    h = entry['height']
    if h < 10:
        ids_0_9.append(uttid)
    elif h <= 15:
        ids_10_15.append(uttid)
    elif h <= 20:
        ids_16_20.append(uttid)
    elif h <= 25:
        ids_21_25.append(uttid)
    elif h <= 30:
        ids_26_30.append(uttid)
    elif h <= 40:
        ids_31_40.append(uttid)
    elif h <= 50:
        ids_41_50.append(uttid)
    else:
        ids_51_up.append(uttid)
        

print("# ids_0_9 = ", len(ids_0_9))
print("# ids_10_15 = ", len(ids_10_15))
print("# ids_16_20 = ", len(ids_16_20))
print("# ids_21_25 = ", len(ids_21_25))
print("# ids_26_30 = ", len(ids_26_30))
print("# ids_31_40 = ", len(ids_31_40))
print("# ids_41_50 = ", len(ids_41_50))
print("# ids_51_up = ", len(ids_51_up))

breakdown = { 'lh 0-9': ids_0_9,
              'lh 10-15': ids_10_15,
              'lh 16-20': ids_16_20,
              'lh 21-25': ids_21_25,
              'lh 26-30': ids_26_30,
              'lh 31-40': ids_31_40,
              'lh 41-50': ids_41_50,
              'lh 51-up': ids_51_up,
          }

print(ids_51_up)
