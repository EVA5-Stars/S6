import matplotlib.pyplot as plt
def plot_stat(stat_list, msg):
    plt.figure(figsize=(20,12))
    plt.plot(stat_list[0],color='Magenta',   label='With L1+BN')
    plt.plot(stat_list[1],color='Yellow',     label='With L2+BN')
    plt.plot(stat_list[2],color='Red',    label='With L1 and L2 with BN')
    plt.plot(stat_list[3],color='Blue',  label='With GBN')
    plt.plot(stat_list[4],color='Green', label='With L1 and L2 with GBN')
        
    plt.xlabel(' Epochs ')

    if msg == 'Loss':
        plt.ylabel(' Loss ')
        plt.title('Total losses vs Epochs')
    elif msg == 'Acc':
        plt.ylabel(' Accuracy ')
        plt.title('Total accuracy vs Epochs')

    plt.legend(loc = 'upper left' , bbox_to_anchor = (1,1))
    plt.legend()
    
    plt.show()
