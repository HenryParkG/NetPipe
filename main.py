import streamlit as st

# Set the title of the app
st.title("Welcome to the NetPipe Application")

# Add a subtitle
st.subheader("Main Page")

# Add some introductory text
st.write("""
This is the main page of the NetPipe application. 
Use the navigation menu to explore different features of the app.
""")

# Add some additional information
st.info("This application is designed to help you manage and analyze network pipelines efficiently.")

# Add an example of a metric
st.metric(label="Active Pipelines", value="5", delta="+2 from last week")

# Add a placeholder for future content
st.write("More features coming soon!")