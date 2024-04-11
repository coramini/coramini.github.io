---
layout: post
title: "【Stable Diffusion weui 源码】从ui开始"
date: 2024-04-07
author: "Cola Liu"
categories: [Diffusion, stable diffusion weui]
usemathjax: true
---
`Stable Diffusion WebUI`（稳定扩散 WebUI）是一个基于 `Stable Diffusion` 的网络用户界面 `（Web User Interface）`。

`Stable Diffusion` 是一种用于生成高质量图像的深度学习模型，其核心思想是通过对潜在空间进行随机游走来生成图像。

`Stable Diffusion WebUI` 可能是一个工具或平台，允许用户通过 Web 界面使用稳定扩散模型进行图像生成和处理。

# 从入口文件开始

首先我们从入口文件开始，`stable-diffusion-webui`的入口文件是`webui.sh`

1、webui.sh
可以看到，`webui.sh` 的 launch脚本是 `launch.py`

<img src="/assets/imgs/ai/diffusion/sdw-001.jpeg" />

2、launch.py
打开`launch.py`,可以看到`main`函数里调用了一个`start()`方法。该方法在`launch_utils.py`中进行定义。

<img src="/assets/imgs/ai/diffusion/sdw-002.jpeg" />

3、modules/launch_utils.py
接着我们来到`launch_utils`中`start`函数的定义，可以看到这里调用了 `webui.webui()` 函数。

<img src="/assets/imgs/ai/diffusion/sdw-003.jpeg" />

4、webui.py
接着打开`webui.py`，看到`webui()`函数。可以看到ui界面的核心定义在`ui.create_ui()`函数中。

其中 `stared.demo.launch()`是`Gradio`的语法（后面会介绍）用于启动我们的 ui server。

<img src="/assets/imgs/ai/diffusion/sdw-004.jpeg" />


5、ui.py

来到 `ui` 的核心文件中，可以看到 `stable diffusion webui`主要用 `Gradio`库进行页面的构建。

`gr.blocks` 就是 `Gradio`封装好的组件。

<img src="/assets/imgs/ai/diffusion/sdw-005.jpeg" />

下面简单介绍一下`stable diffusion webui`中用到的ui库 `Gradio`。

# Gradio
`Gradio`是一个用于构建快速原型和部署机器学习模型的工具。它提供了简单易用的界面，让用户可以轻松地将模型部署为交互式应用程序，`无需深度的前端或后端编程经验`。

下面展示一个最简单的例子 ⬇️

```python
import gradio as gr
import numpy as np
 
def generator(text):
   image = np.ones((100,100,3),np.uint8)
   return image

interface = gr.Interface(fn=generator,inputs="text",outputs="image")
interface.launch(server_port=1234,server_name="127.0.0.1")
```

在上面代码中，我们用`interface`来构建组件。`launch`函数用于启动一个`server`，输入对应的网址可以看到对应的`ui`界面如下。

<img src="/assets/imgs/ai/diffusion/sdw-009.png" />

我们可以对`gr.interface`中的参数进行封装，然后再传入相关的参数 `image_args`，**在`stable diffusion webui`中可以看到很多这种写法**。

```python
image_args = dict(
   fn=generator,
   inputs="text", # input type
   outputs="image" # output type
)
interface = gr.Interface(**image_args)
```


上述例子我们用到原生的`text`和`image`作为输入输出类型。当然，我们可以用`Gradio`封装好的组件进行页面构建。


再来看一个复杂点的例子 🌰

这个例子涉及到`gr.Blocks`、`gr.Markdown`、`gr.Tab`、`gr.Textbox`、`gr.Button`、`gr.Row`、`gr.Image`、`gr.Accordion`等组件。


其中 `text_button.click` 和 `image_button.click` 进行click事件监听绑定。

```python
import numpy as np
import gradio as gr
 
def flip_text(x):
   return x[::-1]
 
def flip_image(x):
   return np.fliplr(x)
 
with gr.Blocks() as demo:
    gr.Markdown("# Flip text or image files using this demo.")
    with gr.Tab("Flip Text"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
            image_button = gr.Button("Flip")
 
    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")
 
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)
 
demo.launch(server_port=8888,server_name="127.0.0.1")
```

在上述例子中，依旧使用`launch`方法来启动我们的`ui server`。

运行结果如下图所示：

<img src="/assets/imgs/ai/diffusion/sdw-010.png" />

<img src="/assets/imgs/ai/diffusion/sdw-011.png" />


回到`stable diffsuion webui`中，在`ui.py`我们可以看到最上面一行ui组件在`ui_toprow.py`中进行定义。其中有三个主要的组件创建方法，`create_prompts`、`create_submit_box`、`create_tools_row`

<img src="/assets/imgs/ai/diffusion/sdw-006.jpeg" />

在 webui界面中分别对应下列的组件 ⬇️。

<img src="/assets/imgs/ai/diffusion/sdw-007.jpeg" />

当我们选择`txt2img`模式时，可以看到`toprow`中`prompt`和`submit（即generate按钮）`所绑定的事件监听如下。

<img src="/assets/imgs/ai/diffusion/sdw-008.jpeg" />

介绍完ui，接下来可以看看具体`txt2img`、`img2img`的运行机制了。