device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(latent_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs in dataloader:
        imgs = imgs.to(device)

        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataset):.4f}")

torch.save(model.state_dict(), "vae_celeba.pth")
print("Model saved as vae_celeba.pth")

