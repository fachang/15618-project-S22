//
//  Network.swift
//  MCNN iOS
//
//  Created by BerthCloud Chou on 2022/3/24.
//

import Foundation
import Metal

public class TestLinearNetwork {
    public typealias DataType = Float32

    private let layers: [NetworkModuleProtocol]
    
    public init() {
        self.layers = [
            LinearLayer(
                nInputFeatures: 3, nOutputFeatures: 2, bias: true, gpu: true),
            /*
            LinearLayer(
                nInputFeatures: 5, nOutputFeatures: 2, bias: true, gpu: true),
             */
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}

public class TestNaiveConvNetwork {
    public typealias DataType = Float32
    
    private let layers: [NetworkModuleProtocol]
    
    public init(gpu: Bool = false) {
        self.layers = [
            Conv2DLayerNaive(nInputChannels: 3, nOutputChannels: 2, bias: true,
                        kernelSize: 3, strideHeight: 2, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}

public class TestImg2colConvNetwork {
    public typealias DataType = Float32
    
    private let layers: [NetworkModuleProtocol]
    
    public init(gpu: Bool = false) {
        self.layers = [
            Conv2DLayerImg2col(nInputChannels: 3, nOutputChannels: 2, bias: true,
                        kernelSize: 3, strideHeight: 2, strideWidth: 1,
                        padding: 1, paddingMode: PaddingMode.zeros, gpu: gpu),
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}

public class TestNaiveBigConvNetwork {
    public typealias DataType = Float32
    
    private let layers: [NetworkModuleProtocol]
    
    public init(gpu: Bool = false) {
        let initKernelsData1: [DataType] = [
            0.130525,0.0503252,-0.158305,0.0885752,-0.13076,-0.1552,0.106448,-0.182584,0.0205052,-0.0684524,-0.183843,0.103792,0.115213,-0.0533879,-0.100657,-0.0263103,-0.0720825,0.0240967,-0.0430632,0.0651566,0.0932871,-0.0727073,-0.0432486,0.00749103,-0.168987,-0.286983,-0.0446803,-0.000490028,0.101758,-0.215287,0.0390758,0.153161,-0.144869,-0.142462,0.00314055,-0.0761714,-0.104034,0.19007,0.0781635,-0.0574874,-0.131684,0.183177,-0.0503963,0.0420198,-0.162612,0.0942378,0.1283,-0.000355067,-0.129837,-0.121634,-0.14702,0.12492,-0.151974,-0.138163,0.0588821,-0.0497169,-0.105288,0.000822362,-0.0828186,0.0555068,-0.258317,0.138207,-0.0277033,0.0788523,-0.234153,-0.271299,0.142731,0.100817,-0.201545,0.0168006,0.0730575,-0.15421,0.0176799,0.113509,-0.106213,0.0615686,-0.0753382,-0.271897,0.399307,8.70894e-05,-0.0798468,-0.145116,-0.36304,0.201701,0.186233,0.019263,0.0680467,-0.0144778,-0.0139892,-0.00694196,0.217922,0.096252,-0.275276,0.00138544,0.0562869,0.10163,-0.145236,-0.10048,0.161545,-0.0887115,0.00598258,0.203584,0.0627076,-0.0532106,-0.16263,-0.195294,-0.0987405,0.0646267,0.00638202,-0.0626393,0.197133,-0.148434,-0.0278494,-0.0105271,-0.00807382,-0.00949981,-0.140354,-0.154309,0.0637039,0.0786443,-0.0487958,0.125786,-0.102269,-0.119854,-0.124938,-0.144895,0.263008,-0.215732,-0.116445,-0.225273,-0.111845,0.232648,0.063283,-0.255081,-0.115782,-0.390181,0.235258,0.237903,-0.107831,-0.100723,-0.100546,0.0394166,-0.0280712,0.177057,-0.0162744,-0.0474184,-0.200661,-0.0583619,0.0838474,0.098379,-0.0808795,0.104057,0.12234,0.0991197,-0.102376,0.00348691,0.120589,0.0908423,-0.119535,-0.0945791,-0.0846573,-0.193398,0.108331,-0.266733,-0.200447,0.126327,-0.198573,-0.016642,0.0630605,0.155481,-0.0386603,-0.188953,-0.217782,-0.17039,0.0168579,0.0617504,0.105833,0.129505,-0.173234,-0.204276,-0.323638,-0.0586106,-0.0459194,-0.0918147,0.12082,-0.095148,-0.0879061,0.0151051,-0.091519,-0.0722888,0.105509,0.0825201,0.0806341,-0.256434,-0.046726,0.110955,-0.146488,-0.0311631,-0.210349,-0.267007,-0.176888,0.141171,0.141339,-0.012289,-0.298704,-0.288022,0.234278,0.0772447,0.115422,-0.372994,-0.414585,0.146901,0.133377,-0.120049,-0.158198,-0.364133,-0.233634,0.0191339,0.0150637,0.123122,-0.266357,-0.163637,0.0719836,0.242157,-0.205995,0.0235058,-0.184116,0.205734,0.312983,0.096152,-0.205795,-0.20176,0.102088,0.101039,0.13216,-0.24151,-0.132973,0.0284921,0.0787319,0.0750887,-0.221881,-0.325301,0.036946,0.0817678,0.242624,0.0256787,-0.19194,-0.130494,0.451405,0.124642,-0.274061,-0.00324158,0.0541123,0.102478,0.241846,-0.184443,-0.201526,0.284053,0.354672,0.190358,-0.27679,-0.0631093,0.197657,-0.0722514,-0.0970067,-0.0605981,0.347603,-0.0814411,-0.323868,-0.281077,0.0791708,0.316736,0.0472461,-0.427353,-0.242309,-0.118353,-0.224665,-0.198594,-0.0173648,0.294541,0.175672,-0.158148,-0.179386,0.0993383,-0.0768011,0.0362934,0.188995,-0.0556387,-0.0700778,-0.00461495,0.164442,0.133341,-0.047807,-0.243399,0.142765,-0.158968,0.0450776,-0.182305,-0.00931315,-0.0815967,0.138664,0.139151,-0.143878,-0.0967609,-0.0141508,0.150604,-0.0708371,0.0811409,0.0653298,-0.205698,-0.150065,-0.0690394,-0.188875,-0.204042,-0.0749361,-0.12762,-0.237951,-0.224151,0.0336662,0.00556468,0.0186824,0.0647728,0.0952912,-0.0921405,0.121876,-0.14882,0.0253134,-0.23134,-0.089263,-0.207959,0.134946,0.0565233,0.179394,0.028495,-0.187267,-0.0701164,0.0349555,0.0630644,0.173084,-0.0173095,0.0904638,-0.0932358,-0.210697,-0.139634,0.145125,0.0547433,-0.0563541,0.155383,0.141047,-0.119529,-0.18814,0.107398,-0.0341357,0.125919,0.0470409,0.0625709,-0.149022,0.131251,0.168586,-0.103341,-0.127528,-0.0539144,0.263851,-0.068345,-0.0739181,-0.12729,-0.0407744,0.0722392,-0.0299488,-0.238446,-0.0691527,0.219495,0.169862,-0.297699,-0.200047,0.0266031,0.0792425,0.146068,-0.225953,0.136116,-0.0233387,0.0991507,-0.0517434,-0.0556125,-0.00610364,0.0547643,-0.195309,-0.236196,-0.179462,-0.203383,-0.0677532,0.032391,-0.00960759,-0.0741488,0.206151,0.125738,0.0654389,-0.0241575,-0.0296723,-0.151122,-0.12212,0.152468,-0.0247585,0.0494641,-0.126441,0.0213185,-0.0647066,0.0855439,-0.11621,0.0123462,-0.124014,0.209002,-0.153415,-0.00964577,0.0311115,0.00566048,0.0932122,-0.141178,-0.118087,0.0230588,0.0267942,-0.235506,-0.0148414,-0.140961,-0.0664163,-0.0101527,0.235345,-0.262726,-0.19121,0.0443624,0.0139312,0.0512082,-0.122315,-0.00520387,-0.214415,-0.245587,0.165714,0.039706,-0.252839,-0.280444,0.0403939,-0.167829,0.207248,0.0602033,0.185475,-0.104926,-0.27385,0.160761,-0.141049,0.10548,-0.254284,-0.107112,0.152969,0.0154989,0.102197,-0.142187,-0.0608457,0.145835,-0.0776624,0.0472867,0.00539817,-0.191237,-0.252423,-0.0918512,-0.17066,0.0669312,-0.0866894,-0.110615,-0.171049,-0.00840975,0.226851,0.0329452,-0.0269713,-0.172823,-0.184486,-0.0801202,-0.167422,-0.240676,-0.237119,-0.17981,0.236082,-0.0846111,-0.00786874,0.0954636,0.158658,0.122976,0.291288,0.0246057,0.0699308,0.0258168,0.0488091,0.220377,-0.0497754,-0.0424167,-0.0936261,-0.453783,-0.169414,0.16276,-0.07928,0.0914674,-0.169138,-0.0732185,0.0323745,-0.119022,-0.0415177,-0.0342918,0.0528811,-0.0447573,-0.0525293,-0.0462932,-0.135653,0.0875741,-0.0172637,0.157086,-0.134794,-0.150572,-0.0122164,-0.0287696,0.117632,-0.155369,0.0836108,0.0219211,-0.0696817,-0.0771439,0.0617451,-0.128658,-0.0423531,0.304759,-0.195575,-0.427545,0.333503,-0.269169,0.174691,0.366595,-0.109209,-0.12234,-0.18263,-0.00792572,0.162912,-0.305958,-0.263727,-0.0601966,-0.170237,-0.12797,0.131467,-0.0660332,-0.0216229,-0.258356,0.0691394,0.0461212,-0.0948333,0.0340345,-0.137957,0.123344,-0.154886,-0.313861,-0.179608,0.108567,0.151248,-0.145368,-0.0956807,-0.2586,-0.0973759,-0.112291,0.0415197,0.00497196,-0.0735136,0.179406,-0.027919,0.0721733,0.0642865,0.0245387,0.0405171,0.12534,-0.0274321,-0.197836,0.196983,-0.071524,0.082691,-0.171806,-0.0968647,0.0416643,0.133389,0.00538848,-0.0226939,0.108064,0.186828,-0.20505,0.0188924,-0.216578,0.0987706,-0.0162423,-0.159905,-0.0213867,-0.192315,0.0666209,-0.190604,-0.00114856,0.0855211,0.197748,-0.278932,-0.16957,0.00426693,0.0248555,-0.171666,-0.195185,-0.0264192,0.0414309,0.137375,-0.0641141,0.184986,-0.202226,0.120566,0.151636,-0.281336,0.0785344,-0.214929,-0.114304,-0.0681752,0.0372322,0.249927,-0.295675,-0.0124645,-0.274951,0.135129,0.0693403,0.0443334,-0.104765,0.120704,-0.0797547,-0.115508,-0.216206,-0.0430345,0.142984,0.0522309,-0.137623,0.0171237,0.0104801,0.169236,-0.175689,0.0270812,-0.171534,0.0170511,-0.0163691,-0.124048,-0.00548646,0.123547,0.154433,-0.12556,-0.107252,-0.134559,-0.122141,-0.0173677,-0.0907837,-0.0698965,0.0554691,-0.0228427,-0.152953,-0.208565,-0.0654322,0.11333,0.192314,0.0634973,0.274443,0.152519,-0.0858842,0.061258,0.0689295,-0.212786,0.0179994,-0.203196,-0.0428443,-0.163151,-0.130251,-0.0805817,-0.238432,-0.276522,-0.318595,-0.331989,-0.0967743,-0.215095,-0.315286,0.100021,0.133432,0.0458147,-0.165504,0.0741797,0.086823,-0.0480163,0.218765,0.0428252,-0.00360429,0.140272,0.0967379,-0.0420984,-0.0877603,0.0656851,-0.0910035,0.0752793,0.088677,0.0792009,0.190326,0.283619,0.080849,0.0780779,0.0875331,-0.183521,0.101088,-0.203195,0.0881009,-0.280056,-0.264958,-0.351025,-0.697477,-0.268016,-0.0180135,-0.100208,0.14086,-0.455139,0.105033,-0.0193781,0.111287,0.105689,-0.144996,0.31782,0.0082691,-0.0416045,-0.278261,0.0243394,0.235931,-0.168592,0.138533,0.0271959,-0.0759489,0.0163452,0.104141,-0.0433369,-0.109069,-0.111001,0.0254265,-0.243742,0.0710236,0.0225475,-0.0309771,-0.214379,0.00931961,0.000218373,0.149738,-0.274768,-0.245194,-0.141394,0.0160751,-0.206744,-0.266306,-0.0486956,-0.28986,-0.187685,-0.231921,0.0817019,0.10288,0.0250887,0.0923379,0.15574,0.0514653,0.139073,-0.0229824,0.0999175,0.0658549,0.220739,0.143517,0.0104875,-0.00202599,0.0228631,0.148545,-0.0109754,0.0720092,-0.0876731,-0.216452,-0.400078,-0.0724495,0.199782,0.155475,0.0369458,-0.0917115,0.21215,-0.111578,0.16475,0.227703,-0.316502,0.1598,0.0969325,0.292276,0.0715007,-0.307374,0.175419,0.134824,0.0705997,-0.0277067,0.164526,0.231381
        ]

        let initBiasData1: [DataType] = [
            -0.2954, -0.0595, -0.1051, -0.2124, -0.3594, -0.4434, -0.2325, -0.1478,
            -0.4652, -0.3166, -0.4775, -0.4244, -0.3072, -0.2732, -0.2939, -0.3443,
            -0.2523, -0.3468, -0.4154, -0.4243, -0.5174, -0.6129, -0.1013, -0.1361,
            -0.3120, -0.2456, -0.3770, -0.3130, -0.3176, -0.3309, -0.3206, -0.2481
        ]

        self.layers = [
            Conv2DLayerNaive(nInputChannels: 1, nOutputChannels: 32, bias: true,
                        kernelSize: 5, strideHeight: 1, strideWidth: 1,
                        padding: 2, paddingMode: PaddingMode.zeros, gpu: gpu,
                        initKernels: initKernelsData1, initBias: initBiasData1),
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}

public class TestImg2colBigConvNetwork {
    public typealias DataType = Float32
    
    private let layers: [NetworkModuleProtocol]
    
    public init(gpu: Bool = false) {
        let initKernelsData1: [DataType] = [
            0.130525,0.0503252,-0.158305,0.0885752,-0.13076,-0.1552,0.106448,-0.182584,0.0205052,-0.0684524,-0.183843,0.103792,0.115213,-0.0533879,-0.100657,-0.0263103,-0.0720825,0.0240967,-0.0430632,0.0651566,0.0932871,-0.0727073,-0.0432486,0.00749103,-0.168987,-0.286983,-0.0446803,-0.000490028,0.101758,-0.215287,0.0390758,0.153161,-0.144869,-0.142462,0.00314055,-0.0761714,-0.104034,0.19007,0.0781635,-0.0574874,-0.131684,0.183177,-0.0503963,0.0420198,-0.162612,0.0942378,0.1283,-0.000355067,-0.129837,-0.121634,-0.14702,0.12492,-0.151974,-0.138163,0.0588821,-0.0497169,-0.105288,0.000822362,-0.0828186,0.0555068,-0.258317,0.138207,-0.0277033,0.0788523,-0.234153,-0.271299,0.142731,0.100817,-0.201545,0.0168006,0.0730575,-0.15421,0.0176799,0.113509,-0.106213,0.0615686,-0.0753382,-0.271897,0.399307,8.70894e-05,-0.0798468,-0.145116,-0.36304,0.201701,0.186233,0.019263,0.0680467,-0.0144778,-0.0139892,-0.00694196,0.217922,0.096252,-0.275276,0.00138544,0.0562869,0.10163,-0.145236,-0.10048,0.161545,-0.0887115,0.00598258,0.203584,0.0627076,-0.0532106,-0.16263,-0.195294,-0.0987405,0.0646267,0.00638202,-0.0626393,0.197133,-0.148434,-0.0278494,-0.0105271,-0.00807382,-0.00949981,-0.140354,-0.154309,0.0637039,0.0786443,-0.0487958,0.125786,-0.102269,-0.119854,-0.124938,-0.144895,0.263008,-0.215732,-0.116445,-0.225273,-0.111845,0.232648,0.063283,-0.255081,-0.115782,-0.390181,0.235258,0.237903,-0.107831,-0.100723,-0.100546,0.0394166,-0.0280712,0.177057,-0.0162744,-0.0474184,-0.200661,-0.0583619,0.0838474,0.098379,-0.0808795,0.104057,0.12234,0.0991197,-0.102376,0.00348691,0.120589,0.0908423,-0.119535,-0.0945791,-0.0846573,-0.193398,0.108331,-0.266733,-0.200447,0.126327,-0.198573,-0.016642,0.0630605,0.155481,-0.0386603,-0.188953,-0.217782,-0.17039,0.0168579,0.0617504,0.105833,0.129505,-0.173234,-0.204276,-0.323638,-0.0586106,-0.0459194,-0.0918147,0.12082,-0.095148,-0.0879061,0.0151051,-0.091519,-0.0722888,0.105509,0.0825201,0.0806341,-0.256434,-0.046726,0.110955,-0.146488,-0.0311631,-0.210349,-0.267007,-0.176888,0.141171,0.141339,-0.012289,-0.298704,-0.288022,0.234278,0.0772447,0.115422,-0.372994,-0.414585,0.146901,0.133377,-0.120049,-0.158198,-0.364133,-0.233634,0.0191339,0.0150637,0.123122,-0.266357,-0.163637,0.0719836,0.242157,-0.205995,0.0235058,-0.184116,0.205734,0.312983,0.096152,-0.205795,-0.20176,0.102088,0.101039,0.13216,-0.24151,-0.132973,0.0284921,0.0787319,0.0750887,-0.221881,-0.325301,0.036946,0.0817678,0.242624,0.0256787,-0.19194,-0.130494,0.451405,0.124642,-0.274061,-0.00324158,0.0541123,0.102478,0.241846,-0.184443,-0.201526,0.284053,0.354672,0.190358,-0.27679,-0.0631093,0.197657,-0.0722514,-0.0970067,-0.0605981,0.347603,-0.0814411,-0.323868,-0.281077,0.0791708,0.316736,0.0472461,-0.427353,-0.242309,-0.118353,-0.224665,-0.198594,-0.0173648,0.294541,0.175672,-0.158148,-0.179386,0.0993383,-0.0768011,0.0362934,0.188995,-0.0556387,-0.0700778,-0.00461495,0.164442,0.133341,-0.047807,-0.243399,0.142765,-0.158968,0.0450776,-0.182305,-0.00931315,-0.0815967,0.138664,0.139151,-0.143878,-0.0967609,-0.0141508,0.150604,-0.0708371,0.0811409,0.0653298,-0.205698,-0.150065,-0.0690394,-0.188875,-0.204042,-0.0749361,-0.12762,-0.237951,-0.224151,0.0336662,0.00556468,0.0186824,0.0647728,0.0952912,-0.0921405,0.121876,-0.14882,0.0253134,-0.23134,-0.089263,-0.207959,0.134946,0.0565233,0.179394,0.028495,-0.187267,-0.0701164,0.0349555,0.0630644,0.173084,-0.0173095,0.0904638,-0.0932358,-0.210697,-0.139634,0.145125,0.0547433,-0.0563541,0.155383,0.141047,-0.119529,-0.18814,0.107398,-0.0341357,0.125919,0.0470409,0.0625709,-0.149022,0.131251,0.168586,-0.103341,-0.127528,-0.0539144,0.263851,-0.068345,-0.0739181,-0.12729,-0.0407744,0.0722392,-0.0299488,-0.238446,-0.0691527,0.219495,0.169862,-0.297699,-0.200047,0.0266031,0.0792425,0.146068,-0.225953,0.136116,-0.0233387,0.0991507,-0.0517434,-0.0556125,-0.00610364,0.0547643,-0.195309,-0.236196,-0.179462,-0.203383,-0.0677532,0.032391,-0.00960759,-0.0741488,0.206151,0.125738,0.0654389,-0.0241575,-0.0296723,-0.151122,-0.12212,0.152468,-0.0247585,0.0494641,-0.126441,0.0213185,-0.0647066,0.0855439,-0.11621,0.0123462,-0.124014,0.209002,-0.153415,-0.00964577,0.0311115,0.00566048,0.0932122,-0.141178,-0.118087,0.0230588,0.0267942,-0.235506,-0.0148414,-0.140961,-0.0664163,-0.0101527,0.235345,-0.262726,-0.19121,0.0443624,0.0139312,0.0512082,-0.122315,-0.00520387,-0.214415,-0.245587,0.165714,0.039706,-0.252839,-0.280444,0.0403939,-0.167829,0.207248,0.0602033,0.185475,-0.104926,-0.27385,0.160761,-0.141049,0.10548,-0.254284,-0.107112,0.152969,0.0154989,0.102197,-0.142187,-0.0608457,0.145835,-0.0776624,0.0472867,0.00539817,-0.191237,-0.252423,-0.0918512,-0.17066,0.0669312,-0.0866894,-0.110615,-0.171049,-0.00840975,0.226851,0.0329452,-0.0269713,-0.172823,-0.184486,-0.0801202,-0.167422,-0.240676,-0.237119,-0.17981,0.236082,-0.0846111,-0.00786874,0.0954636,0.158658,0.122976,0.291288,0.0246057,0.0699308,0.0258168,0.0488091,0.220377,-0.0497754,-0.0424167,-0.0936261,-0.453783,-0.169414,0.16276,-0.07928,0.0914674,-0.169138,-0.0732185,0.0323745,-0.119022,-0.0415177,-0.0342918,0.0528811,-0.0447573,-0.0525293,-0.0462932,-0.135653,0.0875741,-0.0172637,0.157086,-0.134794,-0.150572,-0.0122164,-0.0287696,0.117632,-0.155369,0.0836108,0.0219211,-0.0696817,-0.0771439,0.0617451,-0.128658,-0.0423531,0.304759,-0.195575,-0.427545,0.333503,-0.269169,0.174691,0.366595,-0.109209,-0.12234,-0.18263,-0.00792572,0.162912,-0.305958,-0.263727,-0.0601966,-0.170237,-0.12797,0.131467,-0.0660332,-0.0216229,-0.258356,0.0691394,0.0461212,-0.0948333,0.0340345,-0.137957,0.123344,-0.154886,-0.313861,-0.179608,0.108567,0.151248,-0.145368,-0.0956807,-0.2586,-0.0973759,-0.112291,0.0415197,0.00497196,-0.0735136,0.179406,-0.027919,0.0721733,0.0642865,0.0245387,0.0405171,0.12534,-0.0274321,-0.197836,0.196983,-0.071524,0.082691,-0.171806,-0.0968647,0.0416643,0.133389,0.00538848,-0.0226939,0.108064,0.186828,-0.20505,0.0188924,-0.216578,0.0987706,-0.0162423,-0.159905,-0.0213867,-0.192315,0.0666209,-0.190604,-0.00114856,0.0855211,0.197748,-0.278932,-0.16957,0.00426693,0.0248555,-0.171666,-0.195185,-0.0264192,0.0414309,0.137375,-0.0641141,0.184986,-0.202226,0.120566,0.151636,-0.281336,0.0785344,-0.214929,-0.114304,-0.0681752,0.0372322,0.249927,-0.295675,-0.0124645,-0.274951,0.135129,0.0693403,0.0443334,-0.104765,0.120704,-0.0797547,-0.115508,-0.216206,-0.0430345,0.142984,0.0522309,-0.137623,0.0171237,0.0104801,0.169236,-0.175689,0.0270812,-0.171534,0.0170511,-0.0163691,-0.124048,-0.00548646,0.123547,0.154433,-0.12556,-0.107252,-0.134559,-0.122141,-0.0173677,-0.0907837,-0.0698965,0.0554691,-0.0228427,-0.152953,-0.208565,-0.0654322,0.11333,0.192314,0.0634973,0.274443,0.152519,-0.0858842,0.061258,0.0689295,-0.212786,0.0179994,-0.203196,-0.0428443,-0.163151,-0.130251,-0.0805817,-0.238432,-0.276522,-0.318595,-0.331989,-0.0967743,-0.215095,-0.315286,0.100021,0.133432,0.0458147,-0.165504,0.0741797,0.086823,-0.0480163,0.218765,0.0428252,-0.00360429,0.140272,0.0967379,-0.0420984,-0.0877603,0.0656851,-0.0910035,0.0752793,0.088677,0.0792009,0.190326,0.283619,0.080849,0.0780779,0.0875331,-0.183521,0.101088,-0.203195,0.0881009,-0.280056,-0.264958,-0.351025,-0.697477,-0.268016,-0.0180135,-0.100208,0.14086,-0.455139,0.105033,-0.0193781,0.111287,0.105689,-0.144996,0.31782,0.0082691,-0.0416045,-0.278261,0.0243394,0.235931,-0.168592,0.138533,0.0271959,-0.0759489,0.0163452,0.104141,-0.0433369,-0.109069,-0.111001,0.0254265,-0.243742,0.0710236,0.0225475,-0.0309771,-0.214379,0.00931961,0.000218373,0.149738,-0.274768,-0.245194,-0.141394,0.0160751,-0.206744,-0.266306,-0.0486956,-0.28986,-0.187685,-0.231921,0.0817019,0.10288,0.0250887,0.0923379,0.15574,0.0514653,0.139073,-0.0229824,0.0999175,0.0658549,0.220739,0.143517,0.0104875,-0.00202599,0.0228631,0.148545,-0.0109754,0.0720092,-0.0876731,-0.216452,-0.400078,-0.0724495,0.199782,0.155475,0.0369458,-0.0917115,0.21215,-0.111578,0.16475,0.227703,-0.316502,0.1598,0.0969325,0.292276,0.0715007,-0.307374,0.175419,0.134824,0.0705997,-0.0277067,0.164526,0.231381
        ]

        let initBiasData1: [DataType] = [
            -0.2954, -0.0595, -0.1051, -0.2124, -0.3594, -0.4434, -0.2325, -0.1478,
            -0.4652, -0.3166, -0.4775, -0.4244, -0.3072, -0.2732, -0.2939, -0.3443,
            -0.2523, -0.3468, -0.4154, -0.4243, -0.5174, -0.6129, -0.1013, -0.1361,
            -0.3120, -0.2456, -0.3770, -0.3130, -0.3176, -0.3309, -0.3206, -0.2481
        ]

        self.layers = [
            Conv2DLayerImg2col(nInputChannels: 1, nOutputChannels: 32, bias: true,
                        kernelSize: 5, strideHeight: 1, strideWidth: 1,
                        padding: 2, paddingMode: PaddingMode.zeros, gpu: gpu,
                        initKernels: initKernelsData1, initBias: initBiasData1),
        ];
    }
    
    public func forward(input: Tensor<DataType>) -> Tensor<DataType> {
        var curTensor: Tensor<DataType> = input
        for nnModule in self.layers {
            curTensor = nnModule.forward(input: curTensor)
        }
        return curTensor
    }
}
