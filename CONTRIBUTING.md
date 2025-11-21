# Contributing to IoT-Enabled Adaptive Traffic Control System

Thank you for your interest in contributing to this project!

This system was developed as part of my capstone project at African Leadership University, and I'm excited to collaborate with researchers, engineers, and urban planners to make it better.

---

## **My Vision for This Project**

I built this system to address traffic congestion in Sub-Saharan African cities using affordable, accessible technology. My goals are:

1. **Make it work better** - Improve performance, safety, and reliability
2. **Make it fair** - Extend the reward function to include pedestrians, cyclists, and accessibility
3. **Make it real** - Eventually see this deployed in actual cities to improve people's lives

I welcome contributions that align with these goals!

---

## **How to Contribute**

### **Before You Start:**

If you're planning **significant changes** (new features, architectural changes, reward function modifications), please:

1. **Open a GitHub Issue** first to describe what you want to do
2. **Tag me (@eadewusic)** so I'm notified
3. **Wait for my response** - I'll usually respond within 2-3 days

This helps us:
- Avoid duplicate work
- Ensure changes align with the project's ethical goals
- Discuss the best approach together

### **For Small Changes:**

Bug fixes, documentation improvements, typo corrections → just submit a pull request! No need to ask first.

---

## **Contribution Process**

### **Step 1: Fork the Repository**
```bash
git clone https://github.com/eadewusic/Traffic-Optimization-Capstone-Project/
cd Traffic-Optimization-Capstone-Project
git checkout -b feature/your-feature-name
```

### **Step 2: Make Your Changes**
- Follow the existing code style
- Add comments explaining your changes
- Test thoroughly on Raspberry Pi hardware if possible

### **Step 3: Document Your Changes**
- Update README.md if you add new features
- Add your name to CONTRIBUTORS.md
- Write clear commit messages

### **Step 4: Submit a Pull Request**
- Explain WHAT you changed and WHY
- Reference any related GitHub Issues
- Include test results if applicable

---

## **What We're Looking For**

### **High-Priority Contributions:**

**Multi-Modal Equity Improvements**
- Extending the reward function to include pedestrian delay penalties
- Adding cyclist safety considerations
- Implementing accessibility features (audio signals, extended crossing times)

**Performance Enhancements**
- Reducing sim-to-real performance gap
- Improving Safety Wrapper logic
- Optimizing inference latency

**Hardware Improvements**
- Camera-based vehicle detection to replace buttons
- Better GPIO pin management
- Power failure recovery mechanisms

**Documentation**
- Clearer setup instructions
- Deployment guides for different cities
- Troubleshooting common issues

**Testing & Validation**
- Additional test scenarios
- Multi-intersection coordination
- Long-term stability testing

### **Lower-Priority (But Still Welcome!):**

- Code refactoring for readability
- Performance optimizations
- Additional visualization tools
- Translation to other languages

---

## **What We're NOT Looking For**

- **Removing safety features** - The Safety Wrapper is non-negotiable
- **Surveillance capabilities** - No facial recognition, license plate reading, or individual tracking
- **Commercial closed-source forks** - See licensing section below
- **Changes that violate ethical principles** - See EULA Section 10

---

## **Licensing & Attribution**

### **Your Contributions:**

By contributing to this project, you agree that:

1. **You grant me a license** to use your contributions as part of this project
2. **You retain credit** - Your name will be added to CONTRIBUTORS.md
3. **Your work is original** - You own it or have permission to contribute it
4. **It's open-source** - Your contributions will be under MIT/Apache 2.0 licenses

### **Attribution Requirements:**

All forks, modifications, and derivative works MUST include:

```
Original System Developed by Adewusi Eunice Adebusayo (2025)
African Leadership University
IoT-Enabled Adaptive Traffic Control System using PPO Reinforcement Learning
Research Paper: Sim-to-Real Transfer for Intelligent Traffic Control: Hardware-Aware Policy Optimization with Domain Randomization, Multi-Seed Validation, and Low-Cost Edge Deployment in Sub-Saharan Africa 
```

Please don't remove this attribution or claim the original system as your own work. I put a lot of sleepless nights into this! 😅

---

## **Commercial Use & Deployment**

### **For Research & Education:**
Use freely! Modify, experiment, publish papers. Just cite the original work.

### **For Municipal/Commercial Deployment:**
Please **contact me first** at **euniceadewusic@gmail.com** and cc **e.adewusi@alustudent.com**. We'll discuss:
- Your deployment context and requirements
- Commercial licensing terms (if applicable)
- How I can support your implementation
- Collaboration opportunities

I'm not trying to restrict usage, I just want to:
1. Ensure safe, ethical deployment
2. Learn from real-world implementations
3. Potentially collaborate to make it better
4. Understand how the system performs at scale

---

## **Code of Conduct**

### **Be Respectful:**
- Assume good intentions
- Provide constructive feedback
- Welcome newcomers
- No harassment, discrimination, or toxicity

### **Be Ethical:**
- Don't use this system to harm people
- Don't implement surveillance features
- Don't discriminate against vulnerable road users
- Prioritize safety above all else

### **Be Collaborative:**
- Share your knowledge
- Help others debug
- Document your work
- Give credit where due

---

## **Reporting Bugs**

Found a bug? Please open a GitHub Issue with:

- **Description:** What went wrong?
- **Steps to Reproduce:** How can I see this bug myself?
- **Expected Behavior:** What should have happened?
- **Actual Behavior:** What actually happened?
- **Environment:** Raspberry Pi model, OS version, Python version
- **Logs:** Any error messages or terminal output

---

## **Suggesting Enhancements**

Have an idea? Open a GitHub Issue tagged with `enhancement` and describe:

- **Problem:** What limitation are you addressing?
- **Proposed Solution:** How would you solve it?
- **Alternatives:** Any other approaches considered?
- **Impact:** Who benefits? How does it improve the system?

---

## **Contact Me**

### **For Technical Questions:**
- Open a GitHub Issue (preferred for public discussion)
- Email: euniceadewusic@gmail.com and cc e.adewusi@alustudent.com

### **For Collaboration Proposals:**
- Email: euniceadewusic@gmail.com and cc e.adewusi@alustudent.com
- Subject line: "Collaboration Proposal: [Brief Description]"

### **For Commercial Licensing:**
- Email: Email: euniceadewusic@gmail.com and cc e.adewusi@alustudent.com
- Subject line: "Commercial Licensing Inquiry: [Your Organization]"

I usually respond within 2-3 days. If you don't hear back, please follow up, sometimes emails get lost!

---

## **Thank You!**

Every contribution, whether it's code, documentation, bug reports, or suggestions, helps make this system better and brings us closer to actually deploying it in real cities.

I appreciate you taking the time to collaborate on this project. Together, we can build smarter, more equitable traffic systems for African cities!

**Let's build the future together!** 🚀

---

**Eunice Adewusi Adebusayo**  
African Leadership University  
Kigali, Rwanda  
2025