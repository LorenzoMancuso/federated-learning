import math

x_train = [i for i in range(20)]
print(x_train)
TOTAL_CLIENTS_NUMBER = 4
CLIENT_NUMBER = 4


section_length = math.ceil(len(x_train) / TOTAL_CLIENTS_NUMBER)

starting_index = (section_length) * (CLIENT_NUMBER -1)

ending_index = min(len(x_train), starting_index + section_length) -1


print('total_length: ', len(x_train))
print('section_length: ', section_length)
print(f'{starting_index} : {ending_index}')
print(x_train[starting_index : ending_index + 1])