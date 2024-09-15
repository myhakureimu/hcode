'''
def get_dfa_accuracy(vocab, mode, ys, batch):
    #preds = preds_tv
    #vocab = data_module.vocab
    if mode == 'tv':
        tv_dfas_index = batch['tv_dfas_index']
    dfas = batch['dfas']
    preds = ys.argmax(dim=-1).detach().cpu().numpy()
    if mode == 'tv':
        inputs = batch['xs_tv'].detach().cpu().numpy()
    if mode == 'nm':
        inputs = batch['xs_nm'].detach().cpu().numpy()
    char_labels = []
    
    #total = 0.0
    #correct = 0.0
    for b in range(preds.shape[0]):
        current_labels = []
        # breakpoint()
        
        pred_chars = [
            vocab.get_vocab(token) for token in preds[b]
        ]
        input_chars = [
            vocab.get_vocab(token)
            for token in inputs[b]
            #if self.task.dataset.vocab.get_vocab(token) != "."
        ]
        if mode == 'tv':
            dfa = dfas[tv_dfas_index[b]]
        if mode == 'nm':
            dfa = dfas[b]
            
        preceding = []
        for t in range(len(input_chars)):
            if input_chars[t] not in ['.', '|', '>']:
                preceding.append(input_chars[t])
            if input_chars[t] in ['.', '|']:
                preceding = []
            # take the next prediction
            current_word = " ".join(preceding + [pred_chars[t]])
            #print(preceding, pred_chars[t])
            label = int(dfa(current_word))
            #print(label)
            current_labels.append(label)
            #total += 1
        char_labels.append(current_labels)
    # get the accuracy
    return char_labels #, correct / total
'''
'''
def get_dfa_accuracy(vocab, ys, batch):
    dfas = batch[-1]
    preds = ys.argmax(dim=-1).detach().cpu().numpy()
    inputs = batch[0].detach().cpu().numpy()
    #print('inputs.shape = ',inputs.shape)
    char_labels = []
    total = 0.0
    correct = 0.0
    for b in range(preds.shape[0]):
        current_labels = []
        # breakpoint()
        pred_chars = [
            vocab.get_vocab(token) for token in preds[b]
        ]
        input_chars = [
            vocab.get_vocab(token)
            for token in inputs[b]
            if vocab.get_vocab(token) != "."
        ]
        #print(inputs[b].shape,len(input_chars),np.min(inputs[b]),np.max(inputs[b]))
        #print(len(pred_chars),np.min(preds[b]),np.max(preds[b]))
        dfa = dfas[b]
        for t in range(len(input_chars)):
            if len(input_chars) > t + 1:
                if input_chars[t + 1] in ["|", ">"]:
                    continue
                if input_chars[t + 1] == ".":
                    break
            if len(pred_chars) > t:
                current_chars = input_chars[: t + 1] + [pred_chars[t]]
                # take the last example
                current_word = " ".join(current_chars).split(" | ")[-1]
                
                # lzq
                current_word = " ".join(current_word.split(' > '))
                
                label = int(dfa(current_word))
                if current_word:
                    current_labels.append(label)
                    total += 1
                    correct += label
            else:
                print("preds are shorter than inputs")
                current_labels.append(0)
                total += 1
        char_labels.append(current_labels)
    # get the accuracy
    return char_labels, correct / total
'''
def get_dfa_accuracy(mode, vocab, hatys, batch):
    preds = hatys.argmax(dim=-1).detach().cpu().numpy()
    if mode == 'nm':
        xs = batch['nm_xs'].detach().cpu().numpy()
        ys = batch['nm_ys'].detach().cpu().numpy()
    if mode == 'tv':
        xs = batch['tv_xs'].detach().cpu().numpy()
        ys = batch['tv_ys'].detach().cpu().numpy()
        tv_dfas_index = batch['tv_dfas_index']
    dfas = batch['dfas']
    
    #print('inputs.shape = ',inputs.shape)
    char_labels = []
    total = 0.0
    correct = 0.0
    for b in range(hatys.shape[0]):
        current_labels = []
        # breakpoint()
        pred_chars = [
            vocab.get_vocab(token) for token in preds[b]
        ]
        input_chars = [
            vocab.get_vocab(token)
            for token in xs[b]
            #if vocab.get_vocab(token) != "."
        ]
        label_chars = [
            vocab.get_vocab(token)
            for token in ys[b]
            #if vocab.get_vocab(token) != "."
        ]
        #print(inputs[b].shape,len(input_chars),np.min(inputs[b]),np.max(inputs[b]))
        #print(len(pred_chars),np.min(preds[b]),np.max(preds[b]))
        if mode == 'tv':
            dfa = dfas[tv_dfas_index[b]]
        if mode == 'nm':
            dfa = dfas[b]
        
        preceding = []
        for t in range(len(input_chars)):
            if input_chars[t] not in ['.', ';', '>']:
                preceding.append(input_chars[t])
            if input_chars[t] in ['.', ';']:
                preceding = []
            
            current_word = " ".join(preceding + [pred_chars[t]])
            #print(preceding, pred_chars[t])
            label = int(dfa(current_word))
            current_labels.append(label)
            
            if label_chars[t] not in ['.', ';', '>']:
                correct += label
                total += 1

        char_labels.append(current_labels)
    # get the accuracy
    return char_labels, correct / total