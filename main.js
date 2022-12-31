
//(PC/MOBILE)
const filter = "win16|win32|win64|mac|macintel";
const platform = (filter.indexOf(navigator.platform.toLowerCase()) > 0) ? "PC" : "MOBILE";
console.log(`Client platform : ${platform}`);


//GUI
const w = 300, h = 300;
// const w = 450, h = 330;
const video = document.getElementById("video");
[video.width, video.height] = [w, h];

//WebRTC

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
let backlog;
let vidStream = null;
let cam = "environment" // "environment" or "user"
const successCallback = (_vidStream) => {
    vidStream = _vidStream;
    video.srcObject = _vidStream;
    video.play();
}

const change_cam = () => {
    if (platform == "PC") {
        console.log("Not supported!");
    } else {
        if (vidStream == null) return;
        vidStream.getTracks().forEach(t => {
            t.stop();
        });
        if (cam == "environment") cam = "user";
        else cam = "environment";
        backlog = { video: { facingMode: { exact: cam }, width: { exact: w }, height: { exact: h } }, audio: false };
        navigator.getUserMedia(backlog, successCallback, (e) => console.log(e));
    }
}

if (platform == "PC") {
    backlog = { video: { width: { exact: w }, height: { exact: h } }, audio: false };
} else {
    backlog = { video: { facingMode: { exact: cam }, width: { exact: w }, height: { exact: h } }, audio: false };
}
navigator.getUserMedia(backlog, successCallback, (e) => console.log(e));

const canvas = document.getElementById("canvas");
[canvas.width, canvas.height] = [w, h];
const ctx = canvas.getContext("2d");
ctx.fillStyle = "rgb(0, 255, 0)";
ctx.strokeStyle = "rgb(0, 255, 0)";
ctx.font = "15px Arial";
ctx.lineWidth = 3;

let model;
const load = async () => {
    const url = "./model.json";
    // const url = "model.json";
    model = await tf.loadGraphModel(url);
    console.log("Model loaded");
}
load();

imageDescribe = async () => {
    try {
        canvas.getContext('2d').drawImage(video, 0, 0, 200,200);  
        // save canvas image as byte64 string
        // const img = await canvas.toDataURL().replace(/^data:image\/(png|jpeg);base64,/, '');
        var dataUri = canvas.toDataURL('image/'+"jpeg");
        var d = dataUri.split(',')[1];
        var mimeType = dataUri.split(';')[0].slice(5)
        var bytes = window.atob(d);
        var buf = new ArrayBuffer(bytes.length);
        var byteArr = new Uint8Array(buf);
        for (var i = 0; i < bytes.length; i++) {
            byteArr[i] = bytes.charCodeAt(i);
        }

        // const imageBuffer = Base64
        console.log(byteArr)
        const subscription = '';
        const endpoint = 'https://eastus.api.cognitive.microsoft.com/vision/v2.0/analyze?visualFeatures=Description&language=en';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/octet-stream',
                'Ocp-Apim-Subscription-Key': subscription
            },
            body: byteArr
        });
        const data = await response.json();
        console.log(data);
        console.log(response.status);
        const description = data.description.captions[0].text;
        console.log(description);
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(description));



    } catch (error) {
        console.log(error);

    }
}   



const predict = async () => {
    const image = tf.browser.fromPixels(video).resizeBilinear([224, 224]).transpose([2, 0, 1]).reshape([1, 3, 224, 224]).asType('float32').div(255);
//     const image = tf.browser.fromPixels(video).resizeBilinear([200, 200]).transpose([2, 0, 1]).reshape([1, 3, 200, 200]).asType('float32').div(255);
    result = await model.predict(image);
    console.log("Done");
    
    const outReshape = (tf.transpose(result, [2, 3, 1, 0])).reshape([224, 224, 1]);
    const outResize = tf.mul(tf.div(outReshape, tf.max(outReshape)), 255).asType('int32');
    const tensor = outResize
    const values = tensor.dataSync();
    
    const arr = Array.from(values);

    const resizedArray =tf.image.resizeBilinear(tensor, [10, 10]);
    const resizedValues=resizedArray.dataSync();
    const finalResizedArray = Array.from(resizedValues);
    let arr2d = [];
    let min=256;
    let max=0;
    for (let i = 0; i < 10; i++) {
        arr2d.push(finalResizedArray.slice(i * 10, i * 10 + 10));
        arr2d[i].forEach((item, index) => {
            arr2d[i][index] = Math.floor(item);
            min=Math.min(min,arr2d[i][index]);
            max=Math.max(max,arr2d[i][index]);
        });
    }
    // convert all values in arr2d to 0-255
    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            arr2d[i][j] = Math.floor((arr2d[i][j]-min)/(max-min)*255);
        }
    }
    

    

    var alpha =0
    var beta =0
    var gamma=0
    window.addEventListener('deviceorientation', handleOrientation, true)
      function handleOrientation(event) {
         alpha = event.alpha
         beta = event.beta
         gamma = event.gamma
        //  console log the values
         console.log(alpha, beta, gamma)
        //  get out of the function
        return
      }
    

    // makle a post request to 192.168.29.253/postplain/ with json body of arr2d
    try {
        const response = await fetch('http://192.168.29.253/postplain/', {
            method: 'POST',
            headers: {
                // 'mode': 'no-cors',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"device":depthMatrix,"alpha":alpha,"beta":beta,"gamma":gamma})
        });
        let responseText = await response.text();
        // const data =  await response.text();
        console.log(responseText);
        // decode responseText
        const decoded = JSON.parse(responseText);
        console.log(decoded);
        if (decoded["imageDescription"]==1){
            await imageDescribe();
        }
    
    } catch (error) {
        console.log(error);
    }

    // console.log(typeof arr2d);
    // console.log(arr[0]);
    console.log();
    await tf.browser.toPixels(outResize, canvas);
    setTimeout(predict, 1000);
}

