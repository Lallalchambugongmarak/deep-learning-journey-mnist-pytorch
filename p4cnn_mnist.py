# Day 4: CNN on MNIST
# Final Accuracy: 99.09% after 5 epochs
# Beats Day 3 Linear: 92.26% by +6.83%
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    
    # Test after each epoch
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print(f'Epoch {epoch+1}: {acc:.2f}%')
