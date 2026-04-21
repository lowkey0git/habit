model.eval()
with torch.no_grad():
    z = torch.randn(16, 256).to(device)
    generated = model.decoder(z)

grid = make_grid(generated, nrow=4, normalize=True)
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1,2,0).cpu())
plt.axis("off")
plt.show()
