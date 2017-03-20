import csv
def get_train_data(x_file,t_file):
    with open(x_file, 'r') as locCsvfile:
        loc_reader = csv.reader(locCsvfile)
        train_loc = []
        for row in loc_reader:
            train_loc.append([int(iterator) for iterator in row])
    with open(t_file, 'r') as valCsvfile:
        val_reader = csv.reader(valCsvfile)
        train_val = []
        for row in val_reader:
            train_val.append([int(row[0])])
    return train_loc, train_val # Need to convert to np.array()

def output_test_result(t_file,y):
    with open(t_file, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile)
        for e in y:
            writer.writerow(e)
