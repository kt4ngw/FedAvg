import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getdata import GetDataSet



class Client():
    def __init__(self, trainDataSet, device):
        self.trainDataSet = trainDataSet
        self.device = device
        self.trainDataLoader = None
        self.localPara = None

    def localModelUpdate(self, localEpoch, localBatchSize, model, lossFun, opti, globalPara):


        model.load_state_dict(globalPara, strict=True)

        self.trainDataLoader = DataLoader(self.trainDataSet, batch_size=localBatchSize, shuffle=True)

        for i in range(localEpoch):
            for X, y in self.trainDataLoader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                loss = lossFun(pred, y)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return model.state_dict()




class ClientsGroup():

    def __init__(self, dataSetName, clientsNum, device):

        self.dataSetName = dataSetName
        self.clientsNum = clientsNum
        self.device = device
        self.clients_set = {}
        self.testDataLoader = None
        self.dataSetAllocation()

    def dataSetAllocation(self):

        dataSet = GetDataSet(self.dataSetName, )
        testData = dataSet.testData
        testLabel = dataSet.testLabel

        self.testDataLoader = DataLoader(TensorDataset(testData, testLabel), batch_size=100, shuffle=False)

        trainData = dataSet.trainData
        trainLabel = dataSet.trainLabel


        subClientDataSize = dataSet.trainDataSize // self.clientsNum # z

        for i in range(self.clientsNum):

            localData, localLabel = trainData[i*subClientDataSize:(i+1)*subClientDataSize], trainLabel[i*subClientDataSize:(i+1)*subClientDataSize]

            local = Client(TensorDataset(torch.tensor(localData), torch.tensor(localLabel)), self.device)

            self.clients_set['client{}'.format(i)] = local




