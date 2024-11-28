import csv

def read_and_calculate_average(input_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]

    sum_col3 = sum_col4 = sum_col5 = sum_col6 =0
    for row in data:
        sum_col3 += float(row[2])
        sum_col4 += float(row[3])
        sum_col5 += float(row[4])
        sum_col6 += float(row[5])

    average_col3 = sum_col3 / len(data)
    average_col4 = sum_col4 / len(data)
    average_col5 = sum_col5 / len(data)
    average_col6 = sum_col6 / len(data)
    return header, average_col3, average_col4, average_col5,average_col6

def write_to_csv(output_file, header, average_col3, average_col4, average_col5,average_col6):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header + ['Average'])
        writer.writerow([average_col3, average_col4, average_col5,average_col6])

input_file = '/home/zcchong/jupyterlab/MolSearch-main/MCTS/libs/gsk3b_jnk3_stage1/result_start_mols_task1_seed_0.csv'
output_file = '/home/zcchong/jupyterlab/MolSearch-main/MCTS/libs/gsk3b_jnk3_stage1/2goalsresult_average.csv'
header, average_col3, average_col4, average_col5 ,average_col6= read_and_calculate_average(input_file)
write_to_csv(output_file, header, average_col3, average_col4, average_col5,average_col6)
