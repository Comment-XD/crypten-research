import numpy as np
import copy
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

def soft_R_generate(data, rd_r_list, S, g, prime, label):

    R = []
    Reciver_see = []
    # OUTER LOOP
    
    # For each row of the data,
    # Convert it into 8bit binary
    for i in range(len(data)):
        tmp = ""
        tmp = f'{data[i]:08b}'
        
        
        # 
        tmp_see = [label[int(tmp[0:1], 2)],
                   label[int(tmp[1:2], 2)],
                   label[int(tmp[2:4], 2)],
                   label[int(tmp[4:6], 2)],
                   label[int(tmp[6:8], 2)]]
        
        # Key  
        
        key_list = np.random.randint(0,1,size=(len(rd_r_list),)) #row, coloumn
        tmp_R = np.random.randint(0,1,size=(len(rd_r_list),)) #row, coloumn
        # INNER LOOP
        for i in range(len(rd_r_list)):
            key_tmp = pow(int(g), int(rd_r_list[i]), int(prime))
            key_list[i] = key_tmp
            tmp_R[i] = pow(int(S), int(tmp_see[i]), int(prime)) ^ key_tmp
        Reciver_see.append(tmp_see)
        R.append(list(tmp_R))
    return R, Reciver_see
    # change 1: move key_tmp to OUTER LOOP

def compare (in_j, p_i):
    if in_j < p_i:
        return 1
    elif in_j == p_i:
        return 2
    else:
        return 3
    
    
def Generate_org_massage (Data_send):
    large_massage_matrix = []
    for i in range(len(Data_send)):
        tmp = ""
        tmp = f'{Data_send[i]:08b}'
        tmp_see = [int(tmp[0:1], 2),
                   int(tmp[1:2], 2),
                   int(tmp[2:4], 2),
                   int(tmp[4:6], 2),
                   int(tmp[6:8], 2)]
        tmp_matrix = []
        tmp = []
        for j in range(5):
            if j < 2:
                in_j = tmp_see[j]
                for k in range(2):
                    tmp.append(compare (in_j, k))
            else:
                tmp = []
                in_j = tmp_see[j]
                for k in range(4):
                    tmp.append(compare (in_j, k))
            tmp_matrix.append(tmp)
        large_massage_matrix.append(tmp_matrix[1:5])
    return large_massage_matrix

def key_calculation (label, S, prime, R, x):
    max_length = len(label)
    batch_size = len(R)
    S_label_matrix = np.random.randint(0,1,size=(max_length, 4,)) #row, coloumn
    for j in range(2):
        S_label_matrix[0][j]   = pow(int(S), int(label[j]), int(prime))
        S_label_matrix[0][j+2] = pow(int(S), int(label[j]), int(prime))
    for i in range(1, max_length):
        for j in range(4):
            S_label_matrix[i][j]   = pow(int(S), int(label[j]), int(prime))
    R_matrix = np.random.randint(0,1,size=(batch_size, max_length, 4,)) #row, coloumn
    for i in range(batch_size):
        for j in range(2):
            R_matrix[i][0][j*2]     = R[i][j]
            R_matrix[i][0][j*2 + 1] = R[i][j]
    for i in range(batch_size):
        for k in range(1, max_length):
            for j in range(4):
                R_matrix[i][k][j]     = R[i][k + 1]
    Key_matrix = np.random.randint(0,1,size=(batch_size, max_length, 4,)) #row, coloumn
    for i in range(batch_size):
        for k in range(max_length):
            for j in range(4):
                tmp = R_matrix[i][k][j]^S_label_matrix[k][j]
                Key_matrix[i][k][j] = pow(int(tmp), int(x), int(prime))
    return Key_matrix


def enc_Massage_with_key (massage_matrix, key_matrix):
    enc_Massage_matrix = copy.deepcopy(key_matrix)
    for i in range(len(key_matrix)):
        for k in range(len(key_matrix[0])):
            for j in range(len(key_matrix[0][0])):
                enc_Massage_matrix[i][k][j] = massage_matrix[i][k][j] ^ key_matrix[i][k][j]
    return enc_Massage_matrix

def AQ2DNN(data_size):
    # 1 Generate random number on Sender and Receiver
    g = np.random.randint(0, 255, size=(1,), dtype=np.uint8)[0]
    prime = 255
    label = [1, 2, 3, 4]
    #print("Shared values are:", "g:", g, ", prime:", prime, " and label list: ", label)
    ## Sender
    rd_s = np.random.randint(0, 255, size=(1,), dtype=np.uint8)[0]
    S = pow(int(g), int(rd_s), int(prime))
    #print("Generate random number rd_s in sender side:", rd_s, "Based on this number, we have S:", S)
    #print("Send S to Receiver ... ...")

    # 2 Generate R from the Recivers' data and S
    ## Generate Reciver's data
    Data_rec = np.random.randint(-128, 127, size=(data_size,), dtype=np.int8)
    Data_rec = - Data_rec
    #print("Original Reciver's data is INT8: ", Data_rec)
    Data_rec = np.array(Data_rec, dtype=np.uint8)
    #print("Translate to UINT8: ", Data_rec)

    ## Reciver
    rd_r_list = np.random.randint(0, 255, size=(5,), dtype=np.uint8)
    #print("Generate random number rd_r_list in reciever side:", rd_r_list)
    R, Reciver_see = soft_R_generate(Data_rec, rd_r_list, S, g, prime, label)
    #print("Python generated R:", R, "send to Sender")
    #print("Reciver keep see list it-self for later decoding: ", Reciver_see)

    # 3 Sender use R to generate Massage for Reciever
    ## Generate Sender's data
    Data_send = np.random.randint(-128, 127, size=(data_size,), dtype=np.int8)
    # Data_send = np.array([-74,], dtype=np.int8)
    #print("Original Sender's data is INT8: ", Data_send)
    Data_send = np.array(Data_send, dtype=np.uint8)
    #print("Translate to UINT8: ", Data_send)

    Key_matrix = key_calculation(label, S, prime, R, rd_s)
    Sender_Massage_Matrix = Generate_org_massage(Data_send)
    #rint("The original massage are: ", Sender_Massage_Matrix)
    enc_Massage_matrix = enc_Massage_with_key(Sender_Massage_Matrix, Key_matrix)
    #print("The encrypted massage are: \n", enc_Massage_matrix)

    # 4 Sender use Enc(Ms) to Reciever and generate Mask
    Mask_list = np.random.randint(0, 1, size=(data_size,), dtype=np.uint8)
    inter_matrix = copy.deepcopy(Reciver_see)
    for i in range(len(Reciver_see)):
        idx = label.index(Reciver_see[i][0])
        inter_matrix[i][0] = enc_Massage_matrix[i][0][idx]
        idx = label.index(Reciver_see[i][1])
        inter_matrix[i][1] = enc_Massage_matrix[i][0][idx + 2]
        for j in range(2, len(Reciver_see[0])):
            idx = label.index(Reciver_see[i][j])
            inter_matrix[i][j] = enc_Massage_matrix[i][j - 1][idx]
    de_key = np.random.randint(0, 1, size=(len(rd_r_list),))
    for i in range(len(rd_r_list)):
        de_key[i] = pow(int(S), int(rd_r_list[i]), int(prime))
    #print("Decode key is: ", de_key)
    dems_matrix = copy.deepcopy(Reciver_see)
    for i in range(len(inter_matrix)):
        for j in range(len(inter_matrix[0])):
            dems_matrix[i][j] = inter_matrix[i][j] ^ de_key[j]
    #print("Decrypted interested massage", dems_matrix)
    for i in range(len(dems_matrix)):
        if dems_matrix[i][0] == 3:
            Mask_list[i] = 0
        elif dems_matrix[i][0] == 1:
            Mask_list[i] = 1
        else:
            if Reciver_see[i][0] == label[1]:
                j = 1
                while dems_matrix[i][j] == 2 and j < len(dems_matrix[0]) - 1:
                    j = j + 1
                #print("R,S in,", i + 1, " are negtive and j is", j)
                if dems_matrix[i][j] == 1 or dems_matrix[i][j] == 2:
                    Mask_list[i] = 0
                else:
                    Mask_list[i] = 1
            else:
                j = 1
                while dems_matrix[i][j] == 2 and j < len(dems_matrix[0]) - 1:
                    j = j + 1
                #print("R,S in,", i + 1, " are positive and j is", j)
                if dems_matrix[i][j] == 3:
                    Mask_list[i] = 1
                else:
                    Mask_list[i] = 0

    ground_s = np.array(np.array(Data_send, dtype=np.int8), dtype=np.int16)
    ground_r = np.array(np.array(Data_rec, dtype=np.int8), dtype=np.int16)

    #print("S:", np.array(Data_send, dtype=np.int8))

    #print("R string:\n", R_string)
    #print("Generated mask is: ", Mask_list)
    ground_truth = ground_s - ground_r
    #print("The ground truth is: ", ground_truth)
    ground_Mask_list = copy.deepcopy(Mask_list)
    for i in range(len(ground_truth)):
        if ground_truth[i] > 0:
            ground_Mask_list[i] = 1
        else:
            ground_Mask_list[i] = 0
    Final_R = np.equal(Mask_list, ground_Mask_list)
    # Number_T = np.sum(Final_R == True)
    Number_F = np.sum(Final_R == False)
    if Number_F == 0:
        print("Results Matched for existing RELU")
    #print("Reslut check:", np.equal(Mask_list, ground_Mask_list))
    #print("Number of False is:", Number_F)


def recover_data(original_data,original_positions,shuffled_mask):
    original_mask = np.zeros_like(original_data)
    for i in range(original_data.size):
        original_index = int(original_positions[original_positions[:, i] != -1, i])
        original_mask[original_index] = shuffled_mask[original_positions[:, i] != -1, i]

    return original_mask

def verify_results(mask,data):
    ground_mask = (data > 0).astype(int)
    #print(ground_mask)
    #print(mask)
    return np.array_equal(mask, ground_mask)

def My_Relu(data_size):
    # sender
    Data_send = np.random.randint(-128, 127, size=(data_size,), dtype=np.int8)
    #print(Data_send)
    extended_data_send = np.vstack([Data_send, np.random.randint(-128, 127, (4, Data_send.size), dtype=np.int8)])

    #print(extended_data_send)

    original_positions = np.vstack([np.arange(Data_send.size), np.full((4, Data_send.size), -1)])

    # Shuffle the data within each column, preserving the mapping
    for col in range(extended_data_send.shape[1]):
        combined = np.column_stack((extended_data_send[:, col], original_positions[:, col]))
        np.random.shuffle(combined)
        extended_data_send[:, col], original_positions[:, col] = combined[:, 0], combined[:, 1]

    # Reciever
    Data_rec = np.random.randint(-128, 127, size=(data_size,), dtype=np.int8)
    #print(Data_rec)

    relu_extended_data_send = (extended_data_send + Data_rec).astype(np.int8)
    #print(relu_extended_data_send)
    all_mask = (relu_extended_data_send > 0).astype(int)

    #print(relu_extended_data_send)
    #print(mask)
    # Recover the original 1D array using the positions
    mask = recover_data(Data_send,original_positions,all_mask)

    if verify_results(mask,Data_send + Data_rec):
        print("Results Matched!")


if __name__ == "__main__":
    import time
    ci = 1000  # Number of input channels
    h = 30  # Height of the input feature map,
    w = 20  # Width of the input feature map
    data_size = ci * h * w


    start_time = time.time()
    My_Relu(data_size)
    end_time = time.time()
    execution_time_one = end_time - start_time
    print(f"Execution time of my relu: {execution_time_one:.2f} seconds")

    start_time = time.time()
    AQ2DNN(data_size)
    end_time = time.time()
    execution_time_one = end_time - start_time
    print(f"Execution time of existing relu: {execution_time_one:.2f} seconds")

