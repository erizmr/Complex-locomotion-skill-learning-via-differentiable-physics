for I in 1
do
  rm -rf mass_spring/0.01
  rm -rf mass_spring/0.03
  rm -rf mass_spring/0.07
  python3 mass_spring_multitask.py 5 128
  cd mass_spring/0.01
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../0.01.gif
  cd ../..
  cd mass_spring/0.03
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../0.03.gif
  cd ../..
  cd mass_spring/0.07
  ti video && ti gif -i video.mp4 -f250 && mv video.gif ../0.07.gif
  cd ../..
done
