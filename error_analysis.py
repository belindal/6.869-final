import json
import glob
import os
import numpy as np

trials=5
split="test"
results_dirs = ["vqa_fewshot_pokemon", "vqa_fewshot_pokemon_toks_only_no_vocab", "vqa_fewshot_pokemon_meta_toks_only", "vqa_fewshot_pokemon_anshead_only", "vqa_lxr955"]

overall_accuracy = []
accuracy_trials = []
avg_pos_neg_f1s = []
mut_excl_cm_all = []
rd_to_qtype_to_metrics = {}
for d, rd in enumerate(results_dirs):
    print(rd)
    rd_to_qtype_to_metrics[rd] = {}
    overall_accuracy.append([])
    accuracy_trials.append({})
    avg_pos_neg_f1s.append([])
    mut_excl_cm_trials = []
    for trial in range(trials):
        results_dir = f"snap/vqa/{rd}/{split}_try{trial}_predict"
        # TODO
        score = {}
        total = {}
        overall_score = 0.0
        overall_total = 0
        pos_neg_f1s = []
        # mut_excl_cm = np.array([[0,0],[0,0]])
        mut_excl_cm = np.array([0,0])
        for results_fn in glob.glob(os.path.join(results_dir, "*_human_readable.json")):
            with open(results_fn) as f:
                pred_is_image = set()
                act_is_image = set()
                results = json.load(f)
                for r, result in enumerate(results):
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
                    if 'mutual exclusivity' in q_type:
                        assert result['question']['img_id'] == results[r-1]['question']['img_id']
                        #               mutexcl=left    mutexcl=right
                        # simple=left
                        # simple=right
                        # mut_excl_cm[0 if results[r-1]['answer'] == 'left' else 1][0 if result['answer'] == 'left' else 1] += 1
                        mut_excl_cm[0 if results[r-1]['answer'] == result['answer'] else 1] += 1
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
        avg_pos_neg_f1s[d].append(sum(pos_neg_f1s)/len(pos_neg_f1s))
        mut_excl_cm_trials.append(mut_excl_cm/mut_excl_cm.sum())

        for q_type in score:
            # print(f"{q_type}: {len(accuracy[q_type])/len(total[q_type])}")
            if q_type not in accuracy_trials[d]: accuracy_trials[d][q_type] = []
            accuracy_trials[d][q_type].append(score[q_type]/len(total[q_type]))
        overall_accuracy[d].append(overall_score/overall_total)

    print(f"pos/neg image gen f1s: {sum(avg_pos_neg_f1s[d])/len(avg_pos_neg_f1s[d])*100} +/- {np.std(np.array(avg_pos_neg_f1s[d]))*100}")
    rd_to_qtype_to_metrics[rd]['Image Gen F1'] = {'accuracy': sum(avg_pos_neg_f1s[d])/len(avg_pos_neg_f1s[d])*100, 'stddev': np.std(np.array(avg_pos_neg_f1s[d]))*100}
    for q_type in accuracy_trials[d]:
        avg_acc = sum(accuracy_trials[d][q_type])/len(accuracy_trials[d][q_type]) * 100.0
        stddev_acc = np.std(np.array(accuracy_trials[d][q_type])) * 100.0
        print(f"{q_type}: {avg_acc} +/- {stddev_acc}")
        if 'pos' in q_type or 'neg' in q_type:
            continue
        # if q_type not in rd_to_qtype_to_metrics[rd]:
        rd_to_qtype_to_metrics[rd][q_type] = {'accuracy': avg_acc, 'stddev': stddev_acc}
    overall_accuracy[d] = np.array(overall_accuracy[d])*100
    # import pdb; pdb.set_trace()
    # print(accuracy_trials[d])
    mut_excl_cm_trials = np.stack(mut_excl_cm_trials)
    print(f"Mutual exclusivity confusion matrix:\n{np.mean(mut_excl_cm_trials, axis=0).tolist()}")
    mut_excl_cm_all.append({'average': np.mean(mut_excl_cm_trials, axis=0).tolist()[1], 'stddev': np.std(mut_excl_cm_trials, axis=0).tolist()[1]})
    print(f"{np.sum(overall_accuracy[d])/len(overall_accuracy[d])} +/- {np.std(overall_accuracy[d])}")
    print("====")

print(rd_to_qtype_to_metrics)
print(mut_excl_cm_all)