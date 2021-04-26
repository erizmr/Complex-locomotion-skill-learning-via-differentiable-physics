for I in 1
do
  rm -rf mass_spring/0.01_0.1
  rm -rf mass_spring/0.03_0.1
  rm -rf mass_spring/0.07_0.1
  python3 mass_spring_multitask.py 5 64
  cd mass_spring/0.01_0.1
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../0.01_0.1.gif
  cd ../..
  cd mass_spring/0.03_0.1
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../0.03_0.1.gif
  cd ../..
  cd mass_spring/0.07_0.1
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../0.07_0.1.gif
  cd ../..
done
