

def extract_model_wandb(model):
	conv1_w = model.features[0].weight
	conv1_b = model.features[0].bias
	conv2_w = model.features[2].weight
	conv2_b = model.features[2].bias
	conv3_w = model.features[5].weight
	conv3_b = model.features[5].bias
	conv4_w = model.features[7].weight
	conv4_b = model.features[7].bias
	conv5_w = model.features[10].weight
	conv5_b = model.features[10].bias
	conv6_w = model.features[12].weight
	conv6_b = model.features[12].bias

	linear1_w = model.classifier[0].weight
	linear1_b = model.classifier[0].bias
	linear2_w = model.classifier[2].weight
	linear2_b = model.classifier[2].bias
	linear3_w = model.classifier[4].weight
	linear3_b = model.classifier[4].bias


	print("conv1_w.shape:", conv1_w.shape)
	print("conv1_b.shape:", conv1_b.shape)
	print("conv2_w.shape:", conv2_w.shape)
	print("conv2_b.shape:", conv2_b.shape)
	print("conv3_w.shape:", conv3_w.shape)
	print("conv3_b.shape:", conv3_b.shape)
	print("conv4_w.shape:", conv4_w.shape)
	print("conv4_b.shape:", conv4_b.shape)
	print("conv5_w.shape:", conv5_w.shape)
	print("conv5_b.shape:", conv5_b.shape)
	print("conv6_w.shape:", conv6_w.shape)
	print("conv6_b.shape:", conv6_b.shape)
	print("linear1_w.shape:", linear1_w.shape)
	print("linear1_b.shape:", linear1_b.shape)
	print("linear2_w.shape:", linear2_w.shape)
	print("linear2_b.shape:", linear2_b.shape)
	print("linear3_w.shape:", linear3_w.shape)
	print("linear3_b.shape:", linear3_b.shape)
	conv1_w_np = conv1_w.to(torch.float32).cpu().detach().numpy()
	conv1_b_np = conv1_b.to(torch.float32).cpu().detach().numpy()
	conv2_w_np = conv2_w.to(torch.float32).cpu().detach().numpy()
	conv2_b_np = conv2_b.to(torch.float32).cpu().detach().numpy()
	conv3_w_np = conv3_w.to(torch.float32).cpu().detach().numpy()
	conv3_b_np = conv3_b.to(torch.float32).cpu().detach().numpy()
	conv4_w_np = conv4_w.to(torch.float32).cpu().detach().numpy()
	conv4_b_np = conv4_b.to(torch.float32).cpu().detach().numpy()
	conv5_w_np = conv5_w.to(torch.float32).cpu().detach().numpy()
	conv5_b_np = conv5_b.to(torch.float32).cpu().detach().numpy()
	conv6_w_np = conv6_w.to(torch.float32).cpu().detach().numpy()
	conv6_b_np = conv6_b.to(torch.float32).cpu().detach().numpy()

	linear1_w_np = linear1_w.to(torch.float32).cpu().detach().numpy()
	linear1_b_np = linear1_b.to(torch.float32).cpu().detach().numpy()
	linear2_w_np = linear2_w.to(torch.float32).cpu().detach().numpy()
	linear2_b_np = linear2_b.to(torch.float32).cpu().detach().numpy()
	linear3_w_np = linear3_w.to(torch.float32).cpu().detach().numpy()
	linear3_b_np = linear3_b.to(torch.float32).cpu().detach().numpy()


	conv1_w_np.tofile("./conv1_w_np.bin", sep='')
	conv1_b_np.tofile("./conv1_b_np.bin", sep='')
	conv2_w_np.tofile("./conv2_w_np.bin", sep='')
	conv2_b_np.tofile("./conv2_b_np.bin", sep='')
	conv3_w_np.tofile("./conv3_w_np.bin", sep='')
	conv3_b_np.tofile("./conv3_b_np.bin", sep='')
	conv4_w_np.tofile("./conv4_w_np.bin", sep='')
	conv4_b_np.tofile("./conv4_b_np.bin", sep='')
	conv5_w_np.tofile("./conv5_w_np.bin", sep='')
	conv5_b_np.tofile("./conv5_b_np.bin", sep='')
	conv6_w_np.tofile("./conv6_w_np.bin", sep='')
	conv6_b_np.tofile("./conv6_b_np.bin", sep='')

	linear1_w_np.tofile("./linear1_w_np.bin", sep='')
	linear1_b_np.tofile("./linear1_b_np.bin", sep='')
	linear2_w_np.tofile("./linear2_w_np.bin", sep='')
	linear2_b_np.tofile("./linear2_b_np.bin", sep='')
	linear3_w_np.tofile("./linear3_w_np.bin", sep='')
	linear3_b_np.tofile("./linear3_b_np.bin", sep='')
