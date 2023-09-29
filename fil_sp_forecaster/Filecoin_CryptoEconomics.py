import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.markdown("[![CryptoEconLab](./app/static/cover.png)](https://cryptoeconlab.io)")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
    ### Filecoin Deflationary Supply Explorer
    This web-app uses a digital twin of the Filecoin Economy to explore change in net supply flow, given some user defined assumptions about storage provider behavior.
    
    To learn more about the underlying model assumptions and design, please refer to:
     1. [GitHub repository](https://github.com/protocol/mechafil-jax).
     2. Section 3 of [Agent Based Modeling of the Filecoin Economy](https://arxiv.org/pdf/2307.15200.pdf), published at Chainscience 2023.
    
    ### How to use this app
    
    **👈 Select "Supply Forecaster" from the sidebar** to get started. 
    
    In this page, you can simulate whether future supply will be inflationary or deflationary over the next 12 months. The simulation depends on the configuraiton of:

     - **RB Onboarding Rate** - This is the amount of raw-byte power that is onboarded every day onto the network, in units of PiB/day.
     - **Renewal Rate** - This is the percentage of sectors which are expiring that are renewed.
     - **FIL+ Rate** - This is the percentage of onboarded data that is FIL+.
     - **Av. Sec. Dur.** - This is the average sector duration in days. 
     - **Lock Target** - This is percentage of circulating supply the network targets locking.
     - **Burn Factor** - Factor change increasing or decreasing the current burn rate. 
    
     These values can be configured if you expand the "Macro Configuration" section.
    Once configured, click the "Forecast" button. The digital-twin runs in the background, taking into account both historical data pulled directly from the Filecoin blockchain and the mechanistic laws defining Filecoin's to
    forecast pledge, block rewards, and network power. 

    A second set of configurations are available in the "Optimal Locking" section. Configure the options here to explore the tradeoff between exchange rate sensitivity and costs to illustrate an optimal level of locking to incentivise network growth.

    ### Want to learn more?

    - Check out [CryptoEconLab](https://cryptoeconlab.io)

    - Engage with us on [X](https://x.com/cryptoeconlab)

    - Read more of our research on [Medium](https://medium.com/cryptoeconlab) and [HackMD](https://hackmd.io/@cryptoecon/almanac/)

    ### Disclaimer
    CryptoEconLab designed this application for informational purposes only. CryptoEconLab does not provide legal, tax, financial or investment advice. No party should act in reliance upon, or with the expectation of, any such advice.
"""
)
