function characters = print(set)

for C = 1:25
    characters = reshapeASCII(set(:,C));
    imagesc(characters);
    figure;
end

end