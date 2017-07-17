function C = BuildConfusionMatrix_spec(classification, ground_truth, cat_order)
%% similar to BuildConfusionMatrix but the confmat (C) is constructed using
%% cat_order
%% for classification generalisation test

NCATS = length(cat_order);
assert(NCATS == length(unique(ground_truth)), 'Error: categories mismatch');
C=zeros(NCATS,NCATS);

for gt=1:NCATS
    clsidx=classification(ground_truth==cat_order(gt));
    for cls=1:NCATS
        C(gt,cls)=sum(clsidx==cat_order(cls));        
    end
    %C(gt,:)=C(gt,:)./norm(C(gt,:));
    C(gt,:)=C(gt,:)./sum(C(gt,:));
end
end