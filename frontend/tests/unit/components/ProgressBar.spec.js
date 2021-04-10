import { shallowMount } from "@vue/test-utils";
import ProgressBar from "@/components/ProgressBar.vue";

describe("ProgressBar.vue", () => {
  it("renders props.msg when passed", () => {
    const msg = "new message";
    const wrapper = shallowMount(ProgressBar, {
      props: { msg },
    });
    expect(wrapper).toMatchSnapshot();
  });
});
