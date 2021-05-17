import json
import glob
import os
import numpy as np

trials=5

overall_accuracy = []
accuracy_trials = {}
for trial in range(trials):
    # results_dir = f"snap/vqa/vqa_fewshot_pokemon_meta_toks_only/val_try{trial}_predict"
    results_dir = f"snap/vqa/vqa_lxr955/val_try{trial}_predict"
    score = {}
    total = {}
    for results_fn in glob.glob(os.path.join(results_dir, "*_human_readable.json")):
        with open(results_fn) as f:
            results = json.load(f)
            for result in results:
                q_type = result['question']['question_id']
                if q_type not in score:
                    score[q_type] = 0
                    total[q_type] = []
                score[q_type] += result['question']['label'].get(result['answer'], 0.0)
                total[q_type].append(result)

    for q_type in score:
        # print(f"{q_type}: {len(accuracy[q_type])/len(total[q_type])}")
        if q_type not in accuracy_trials: accuracy_trials[q_type] = []
        accuracy_trials[q_type].append(score[q_type]/len(total[q_type]))
        overall_accuracy.append(score[q_type]/len(total[q_type]))

for q_type in accuracy_trials:
    print(f"{q_type}: {sum(accuracy_trials[q_type])/len(accuracy_trials[q_type]) * 100.0} +/- {np.std(np.array(accuracy_trials[q_type])) * 100.0}")
overall_accuracy = np.array(overall_accuracy)*100
# import pdb; pdb.set_trace()
# print(overall_accuracy)
print(f"{np.sum(overall_accuracy)/len(overall_accuracy)} +/- {np.std(overall_accuracy)}")