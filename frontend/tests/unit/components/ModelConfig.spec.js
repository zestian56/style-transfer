import { shallowMount } from "@vue/test-utils";
import ModelConfig from "@/components/ModelConfig.vue";

describe("ModelConfig.vue", () => {
  it("renders props.msg when passed", () => {
    const msg = "new message";
    const wrapper = shallowMount(ModelConfig, {
      props: { msg },
    });
    expect(wrapper).toMatchSnapshot();
  });
});
