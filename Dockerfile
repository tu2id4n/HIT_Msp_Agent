FROM aceracer/hit_pmm_fix:v1
EXPOSE 10080
ENV NAME Agent
# Run app.py when the container launches
WORKDIR /home/pmm
ENTRYPOINT ["python"]
CMD ["run.py"]
