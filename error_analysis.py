import json

results_fn = "snap/vqa/vqa_lxr955_results/minival_predict_human_readable.json"
errors = []
with open(results_fn) as f:
    results = json.load(f)
    for result in results:
        if result['answer'] not in result['question']['label']:
            errors.append(result)

print(len(errors))
for i in range(50):
    print(errors[i])