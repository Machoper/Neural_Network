function combined = decode(W)
combined = W(:,1);
for C = 2:10
    combined = combined + W(:,C);
end
imagesc(reshapeASCII(combined));
end