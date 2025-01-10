from networks.unet import Traditional_U_Net

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = Traditional_U_Net(in_ch=in_chns, out_ch=class_num)

    else:
        net = None
    print('Using the {} model'.format(net_type))
    
    return net

