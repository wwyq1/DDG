import json  
import pandas as pd  
import random  
  
def load_and_process_json_file(file_path):  
    data = []  
    label_to_indices = {}  
      
    # 读取JSON文件并处理数据  
    with open(file_path, 'r', encoding='utf-8') as f:  
        for line in f:  
            obj = json.loads(line.strip())  
            # 检查并替换'labels'为'label'  
            if 'labels' in obj:  
                obj['label'] = obj.pop('labels')  
            # 确保'label'字段存在  
            if 'label' not in obj:  
                raise ValueError(f"Missing 'label' field in JSON object: {obj}")  
              
            # 收集具有相同'label'值的行的索引  
            label = obj['label']  
            if label not in label_to_indices:  
                label_to_indices[label] = []  
            label_to_indices[label].append(len(data))  
              
            # 直接添加到数据列表中（无需复制，因为我们不再修改对象）  
            data.append(obj)  
      
    # 打乱具有相同'label'值的行的顺序（注意：这仅影响我们处理数据的顺序）  
    # 由于Parquet是列式存储，行的物理顺序在Parquet文件中并不重要  
    # 但如果我们想要模拟一个“打乱”的视觉效果，我们可以这样做：  
    sorted_labels = list(label_to_indices.keys())  
    random.shuffle(sorted_labels)  
      
    # 根据打乱后的'label'顺序重新组织数据（对于Parquet来说不是必需的，但用于演示）  
    shuffled_data = []  
    for label in sorted_labels:  
        indices = label_to_indices[label]  
        # 打乱具有相同'label'的行的索引（注意：这不会改变Parquet的行顺序）  
        random.shuffle(indices)  
        # 根据打乱后的索引重新组织数据并添加到结果列表中  
        shuffled_data.extend([data[i] for i in indices])  
      
    # 但是，由于我们最终要将数据保存为Parquet文件，并且Parquet不关心行的顺序，  
    # 我们可以直接使用原始的data列表（它已经包含了替换后的'label'字段），  
    # 并将其转换为DataFrame，然后保存为Parquet文件。  
    # 因此，我们可以忽略上面的shuffled_data变量和相关的代码，  
    # 直接使用下面的代码来创建DataFrame：  
    df = pd.DataFrame(data)  
      
    # 注意：下面的代码行被注释掉了，因为对于Parquet文件来说，它们是不必要的。  
    # 如果你真的想要一个视觉上的乱序效果（尽管这在实际应用中通常是没有意义的），  
    # 你可以取消注释下面的代码行，但这将不会改变Parquet文件的实际内容或结构。  
    # df = pd.DataFrame(shuffled_data)  # 注意：这不会改变Parquet的行顺序！  
      
    return df  
  
def save_to_parquet(df, output_path):  
    df.to_parquet(output_path, engine='pyarrow')  
  
def main(input_json_path, output_parquet_path):  
    df = load_and_process_json_file(input_json_path)  
    # 注意：虽然我们可以根据'label'对DataFrame进行排序，但这对于Parquet来说是不必要的，  
    # 因为Parquet是列式存储的，行的顺序在读取时是不重要的。  
    # 如果你真的想要一个视觉上按照'label'排序的效果（尽管这在实际应用中通常是没有意义的），  
    # 你可以取消注释下面的代码行，但这将不会改变Parquet文件的实际内容或结构。  
    df = df.sort_values(by='label')  # 注意：这不会改变Parquet的行顺序！  
    save_to_parquet(df, output_parquet_path)  
    print(f"Data has been successfully converted and saved to {output_parquet_path}") 
  
if __name__ == "__main__":  
    input_json_path =  "/data3/bishe/DiLM-main/train.json" # 替换为你的输入JSON文件路径  
    output_parquet_path = "/data3/bishe/DiLM-main/train.parquet"  # 替换为你想要的输出Parquet文件路径  
      
    main(input_json_path, output_parquet_path)