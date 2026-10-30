# The README's demo: a simulated Blackhole chip in a container.
#   vhs docs/demo.tape          (CI records it: .github/workflows/demo.yml)
Output docs/img/demo.gif

Require docker

Set Shell "bash"
Set FontSize 15
Set Width 1200
Set Height 620
Set Padding 20
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Type "docker run --rm -e TENSIXFUSE_ARCH=blackhole ghcr.io/tanaymihani/tensixfuse tensixfuse-doctor"
Sleep 400ms
Enter
Wait@300s /\$\s*$/
Sleep 3s

Type "docker run --rm -e TENSIXFUSE_ARCH=blackhole ghcr.io/tanaymihani/tensixfuse pytest -q -m ttsim tests/test_ttsim.py"
Sleep 400ms
Enter
Wait@600s /\$\s*$/
Sleep 4s
