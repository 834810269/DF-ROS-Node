#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from models.flow_net import  Unflow_correlation
Backward_tensorGrid = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
    # end

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensorInput,
                                           grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2,
                                                                                                                   3,
                                                                                                                   1),
                                           mode='bilinear', padding_mode='border')


# end

##########################################################

class Unflow(torch.nn.Module):
    def __init__(self):
        super(Unflow, self).__init__()

        class Upconv(torch.nn.Module):
            def __init__(self):
                super(Upconv, self).__init__()

                self.moduleSixOut = torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, stride=1,
                                                    padding=1)

                self.moduleSixUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                            padding=1)

                self.moduleFivNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFivOut = torch.nn.Conv2d(in_channels=1026, out_channels=2, kernel_size=3, stride=1,
                                                    padding=1)

                self.moduleFivUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                            padding=1)

                self.moduleFouNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=1026, out_channels=256, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFouOut = torch.nn.Conv2d(in_channels=770, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleFouUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                            padding=1)

                self.moduleThrNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=770, out_channels=128, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThrOut = torch.nn.Conv2d(in_channels=386, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleThrUp = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2,
                                                            padding=1)

                self.moduleTwoNext = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=386, out_channels=64, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwoOut = torch.nn.Conv2d(in_channels=194, out_channels=2, kernel_size=3, stride=1, padding=1)

                self.moduleUpscale = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1,
                                             bias=False),
                    torch.nn.ReplicationPad2d(padding=[0, 1, 0, 1])
                )

            # end

            def forward(self, tensorFirst, tensorSecond, objectInput):
                objectOutput = {}

                tensorInput = objectInput['conv6']
                objectOutput['flow6'] = self.moduleSixOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput['conv5'], self.moduleFivNext(tensorInput), self.moduleSixUp(objectOutput['flow6'])], 1)
                objectOutput['flow5'] = self.moduleFivOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput['conv4'], self.moduleFouNext(tensorInput), self.moduleFivUp(objectOutput['flow5'])], 1)
                objectOutput['flow4'] = self.moduleFouOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput['conv3'], self.moduleThrNext(tensorInput), self.moduleFouUp(objectOutput['flow4'])], 1)
                objectOutput['flow3'] = self.moduleThrOut(tensorInput)
                tensorInput = torch.cat(
                    [objectInput['conv2'], self.moduleTwoNext(tensorInput), self.moduleThrUp(objectOutput['flow3'])], 1)
                objectOutput['flow2'] = self.moduleTwoOut(tensorInput)

                return self.moduleUpscale(self.moduleUpscale(objectOutput['flow2'])) * 20.0

        # end
        # end

        class Complex(torch.nn.Module):
            def __init__(self):
                super(Complex, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([2, 4, 2, 4]),
                    torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleRedir = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleCorrelation = Unflow_correlation.ModuleCorrelation()

                self.moduleCombined = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=473, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleUpconv = Upconv()

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFlow):
                objectOutput = {}

                assert (tensorFlow is None)

                objectOutput['conv1'] = self.moduleOne(tensorFirst)
                objectOutput['conv2'] = self.moduleTwo(objectOutput['conv1'])
                objectOutput['conv3'] = self.moduleThr(objectOutput['conv2'])

                tensorRedir = self.moduleRedir(objectOutput['conv3'])
                tensorOther = self.moduleThr(self.moduleTwo(self.moduleOne(tensorSecond)))
                tensorCorr = self.moduleCorrelation(objectOutput['conv3'], tensorOther)

                objectOutput['conv3'] = self.moduleCombined(torch.cat([tensorRedir, tensorCorr], 1))
                objectOutput['conv4'] = self.moduleFou(objectOutput['conv3'])
                objectOutput['conv5'] = self.moduleFiv(objectOutput['conv4'])
                objectOutput['conv6'] = self.moduleSix(objectOutput['conv5'])

                return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)

        # end
        # end

        class Simple(torch.nn.Module):
            def __init__(self):
                super(Simple, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([2, 4, 2, 4]),
                    torch.nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([1, 3, 1, 3]),
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.ZeroPad2d([0, 2, 0, 2]),
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=0),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleUpconv = Upconv()

            # end

            def forward(self, tensorFirst, tensorSecond, tensorFlow):
                objectOutput = {}

                tensorWarp = Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow)

                objectOutput['conv1'] = self.moduleOne(
                    torch.cat([tensorFirst, tensorSecond, tensorFlow, tensorWarp, (tensorFirst - tensorWarp).abs()], 1))
                objectOutput['conv2'] = self.moduleTwo(objectOutput['conv1'])
                objectOutput['conv3'] = self.moduleThr(objectOutput['conv2'])
                objectOutput['conv4'] = self.moduleFou(objectOutput['conv3'])
                objectOutput['conv5'] = self.moduleFiv(objectOutput['conv4'])
                objectOutput['conv6'] = self.moduleSix(objectOutput['conv5'])

                return self.moduleUpconv(tensorFirst, tensorSecond, objectOutput)

        # end
        # end

        self.moduleFlownets = torch.nn.ModuleList([
            Complex(),
            Simple(),
            Simple()
        ])


    # end
    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                # # m.weight.data.normal_(0, 0.02)
                m.weight.data = torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = tensorFirst[:, [2, 1, 0], :, :]
        tensorSecond = tensorSecond[:, [2, 1, 0], :, :]

        tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - (104.920005 / 255.0)
        tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - (110.175300 / 255.0)
        tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - (114.785955 / 255.0)

        tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - (104.920005 / 255.0)
        tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - (110.175300 / 255.0)
        tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - (114.785955 / 255.0)

        tensorFlow = None

        for moduleFlownet in self.moduleFlownets:
            tensorFlow = moduleFlownet(tensorFirst, tensorSecond, tensorFlow)
        # end

        return [tensorFlow]


# end
# end

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = Unflow().to(device)
net.init_weights()
moduleNetwork = net.eval()

def estimate(tensorFirst, tensorSecond):
    #assert(tensorFirst.size(1) == tensorSecond.size(1))
    #assert(tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(3)
    intHeight = tensorFirst.size(2)

    #assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    #assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tensorPreprocessedFirst = tensorFirst.cuda()#.view(1, 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda()#.view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    print(intHeight)
    print(intWidth)
    print(intPreprocessedHeight)
    print(intPreprocessedWidth)

    tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tensorout = moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)
    print(tensorout.size())
    tensorFlow = torch.nn.functional.interpolate(input=tensorout, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tensorFlow[0, :, :, :]
# end

##########################################################

if __name__ == '__main__':
    tensorFirst = torch.randn(4,3,128,416)
    tensorSecond = torch.randn(4,3,128,416)

    tensorOutput = estimate(tensorFirst, tensorSecond)
    print(tensorOutput.size())
