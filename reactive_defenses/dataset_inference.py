## Anonynmous Code File
## Dataset Inference @ ICLR2021: Blind Walk attack


def blind_walk(model, X, y, args):
    #This function generates a single feature for the embedding matrix via Blind Walk Attack
    start = time.time()
    #params for noise magnitudes
    uni, std, scale = (0.005, 0.005, 0.01)
    #Gaussian Noise Sampler
    noise_2 = lambda X: torch.normal(0, std, size=X.shape).cuda()
    #Laplacian Noise Sampler
    noise_1 = lambda X: torch.from_numpy(np.random.laplace(loc=0.0, scale=scale, size=X.shape)).float().to(X.device)
    #Uniform Noise Sampler
    noise_inf = lambda X: torch.empty_like(X).uniform_(-uni,uni)

    noise_map = {"l1":noise_inf, "l2":noise_2, "linf":noise_inf}
    #initialize at x+\delta_p
    mag = 1

    delta = noise_map[args.distance](X)
    delta_base = delta.clone()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)
    with torch.no_grad(): #gradient free
        for t in range(num_steps):
            if t>0:
                preds = model(X_r+delta_r)
                new_remaining = (preds.max(1)[1] == y[remaining])
                remaining[remaining] = new_remaining
            else:
                preds = model(X+delta)
                remaining = (preds.max(1)[1] == y)

            if remaining.sum() == 0: break

            #Only query the data points that have still not reached their neighbour (save queries :)
            X_r = X[remaining]; delta_r = delta[remaining]
            preds = model(X_r + delta_r)
            ## Move by one more step in the same initial direction for the points still in their true class
            mag+=1; delta_r = delta_base[remaining]*mag
            # clip X+delta_r[remaining] to [0,1]
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1-X_r)
            delta[remaining] = delta_r.detach()

        print(f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]==y).sum().item()} | Time taken = {time.time() - start}")
    if is_training:
        model.train()
    return delta


def get_label_only_blind_walk_embeddings(args, loader, model, num_images = 1000):
    ## This function is used to create the output embedding (size 30) via the blind walk attack
    print("Getting Blind Walk Embeddings")
    batch_size = args.batch_size
    max_iter = num_images/batch_size
    lp_dist = [[],[],[]]
    for i,batch in enumerate(loader):
        for j,distance in enumerate(["linf", "l2", "l1"]):
            #linf distance corresponds to uniform noise
            #l2 distance corresponds to gaussian noise
            #l1 distance corresponds to laplacian noise
            temp_list = []
            for target_i in range(10):
                #10 random starts for 10*3 = 30 size embedding
                X,y = batch[0].to(device), batch[1].to(device)
                args.distance = distance
                preds = model(X)
                delta = blind_walk(model, X, y, args) #get one perturbation via blind walk
                distance_dict = {"linf": norms_linf_squeezed, "l1": norms_l1_squeezed, "l2": norms_l2_squeezed}
                distances = distance_dict[distance](delta) ## Distance of perturbation is the feature in the embedding
                temp_list.append(distances.cpu().detach().unsqueeze(-1))
            temp_dist = torch.cat(temp_list, dim = 1)
            lp_dist[j].append(temp_dist)
        if i+1>=max_iter:
            break
    # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
    lp_d = [torch.cat(lp_dist[i], dim = 0).unsqueeze(-1) for i in range(3)]
    # full_d = [num_images, num_classes, num_attacks]
    full_d = torch.cat(lp_d, dim = -1); print(full_d.shape)

    return full_d