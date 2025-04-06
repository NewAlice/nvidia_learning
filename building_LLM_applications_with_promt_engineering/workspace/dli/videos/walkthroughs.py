from IPython.display import HTML

def walkthrough_00():    
    video_url = "https://d36m44n9vdbmda.cloudfront.net/assets/s-fx-12-v2/walkthrough_00.mp4"
    
    video_html = f"""
    <video controls width="640" height="360">
        <source src="{video_url}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    
    display(HTML(video_html))