import json
import glob
import os
import numpy as np

trials=5

overall_accuracy = []
accuracy_trials = {}
avg_pos_neg_f1s = []
for trial in range(trials):
    #results_dir = f"snap/vqa/vqa_fewshot_pokemon_add_toks/val_try{trial}_predict"
    results_dir = f"snap/vqa/vqa_fewshot_pokemon/test_try{trial}_predict"
    #results_dir = f"snap/vqa/vqa_fewshot_pokemon_toks_only_no_vocab/test_try{trial}_predict"
    #results_dir = f"snap/vqa/vqa_fewshot_pokemon_meta_toks_only/test_try{trial}_predict"
    #results_dir = f"snap/vqa/vqa_fewshot_pokemon_anshead_only/test_try{trial}_predict"
    #results_dir = f"snap/vqa/vqa_lxr955/test_try{trial}_predict"
    # TODO
    score = {}
    total = {}
    overall_score = 0.0
    overall_total = 0
    pos_neg_f1s = []
    for results_fn in glob.glob(os.path.join(results_dir, "*_human_readable.json")):
        with open(results_fn) as f:
            pred_is_image = set()
            act_is_image = set()
            results = json.load(f)
            for result in results:
                #import pdb; pdb.set_trace()
                q_type = result['question']['question_type']
                if q_type == 'pos, image gen' or q_type == 'neg, image gen':
                    if result['answer'] == 'yes':
                        pred_is_image.add(result['question']['img_id'])
                    else: 
                        if not result['answer'] == 'no':
                            # always wrong, so do opposite of what is correct
                            if q_type.startswith("neg"):
                                pred_is_image.add(result['question']['img_id'])
                    if q_type.startswith("pos"):
                        act_is_image.add(result['question']['img_id'])
                if q_type not in score:
                    score[q_type] = 0
                    total[q_type] = []
                score[q_type] += result['question']['label'].get(result['answer'], 0.0)
                total[q_type].append(result)
                overall_score += result['question']['label'].get(result['answer'], 0.0)
                overall_total += 1
            correct_is_image = pred_is_image.intersection(act_is_image)
            if len(correct_is_image) == 0:
                pos_neg_f1s.append(0)
            else:
                p, r = len(correct_is_image)/len(pred_is_image), len(correct_is_image)/len(act_is_image)
                pos_neg_f1s.append(2*p*r/(p+r))
    if len(pos_neg_f1s) == 0: import pdb ;pdb.set_trace()
    avg_pos_neg_f1s.append(sum(pos_neg_f1s)/len(pos_neg_f1s))

    for q_type in score:
        # print(f"{q_type}: {len(accuracy[q_type])/len(total[q_type])}")
        if q_type not in accuracy_trials: accuracy_trials[q_type] = []
        accuracy_trials[q_type].append(score[q_type]/len(total[q_type]))
    overall_accuracy.append(overall_score/overall_total)

print(f"pos/neg image gen f1s: {sum(avg_pos_neg_f1s)/len(avg_pos_neg_f1s)*100} +/- {np.std(np.array(avg_pos_neg_f1s))*100}")
for q_type in accuracy_trials:
    print(f"{q_type}: {sum(accuracy_trials[q_type])/len(accuracy_trials[q_type]) * 100.0} +/- {np.std(np.array(accuracy_trials[q_type])) * 100.0}")
overall_accuracy = np.array(overall_accuracy)*100
# import pdb; pdb.set_trace()
# print(accuracy_trials)
print(f"{np.sum(overall_accuracy)/len(overall_accuracy)} +/- {np.std(overall_accuracy)}")
